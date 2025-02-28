"""Functions used to fulfill the add_documents endpoint"""
import concurrent
import copy
import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import ContextManager

import PIL
import ffmpeg
import numpy as np
import torch
from PIL.ImageFile import ImageFile

import marqo.exceptions as base_exceptions
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import *
from marqo.exceptions import InternalError
from marqo.s2_inference import clip_utils
from marqo.s2_inference.errors import UnsupportedModalityError, S2InferenceError, MediaMismatchError, MediaDownloadError, MediaExceedsMaxSizeError
from marqo.s2_inference.models.model_type import ModelType
from marqo.s2_inference.s2_inference import is_preprocessor_preload, load_multimodal_model_and_get_preprocessors, \
    infer_modality, Modality
from marqo.tensor_search.utils import read_env_vars_and_defaults_ints
from marqo.tensor_search import enums
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.models.preprocessors_model import Preprocessors
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.streaming_media_processor import StreamingMediaProcessor
from marqo.tensor_search.telemetry import RequestMetricsStore, RequestMetrics

logger = logging.getLogger(__name__)


def threaded_download_and_preprocess_content(allocated_docs: List[dict],
                                             media_repo: dict,
                                             tensor_fields: List[str],
                                             device: str = None,
                                             media_field_types_mapping: Optional[Dict[str, FieldType]] = None,
                                             media_download_headers: Optional[Dict] = None,
                                             metric_obj: Optional[RequestMetrics] = None,
                                             preprocessors: Optional[Preprocessors] = None,
                                             marqo_index_type: Optional[IndexType] = None,
                                             marqo_index_model: Optional[Model] = None,
                                             audio_preprocessing: Optional[AudioPreProcessing] = None,
                                             video_preprocessing: Optional[VideoPreProcessing] = None,
                                             force_download: bool = False) -> None:
    """A thread calls this function to download images for its allocated documents

    This should be called only if treat URLs as images is True.

    Args:
        allocated_docs: docs with images to be downloaded by this thread,
        media_repo: dictionary that will be mutated by this thread. It will add media
            as values and the URLs as keys
        tensor_fields: A tuple of tensor_fields. Images will be downloaded for these fields only.
        media_download_headers: A dict of headers for image download. Can be used
            to authenticate image downloads
        force_download: If True, skip the _is_image check and download the fields as images.
    Side Effects:
        Adds members to the image_repo dict. Each key is a string which is identified as a URL.
        Each value is either a PIL image, or UnidentifiedImageError, if there were any errors encountered retrieving
        the image.
        For example:
        {
            'https://google.com/my_dog.png': UnidentifiedImageError, # error because such an image doesn't exist
            'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png': <PIL image>
        }
    Returns:
        None

    """
    # Determine index type
    is_structured_index = marqo_index_type == IndexType.Structured
    is_unstructured_index = marqo_index_type in [IndexType.Unstructured, IndexType.SemiStructured]
    # Generate pseudo-unique ID for thread metrics.
    _id = f'image_download.{threading.get_ident()}'
    TIMEOUT_SECONDS = 3
    if metric_obj is None:  # Occurs predominately in testing.
        metric_obj = RequestMetricsStore.for_request()
        RequestMetricsStore.set_in_request(metrics=metric_obj)

    # For backward compatibility, we should accept both supportedModalities and supported_modalities
    if marqo_index_model.properties.get("supported_modalities") and marqo_index_model.properties.get("supportedModalities"):
        raise InvalidModelPropertiesError(
            "Model properties must have either 'supported_modalities' or 'supportedModalities', not both"
        )
    else:
        supported_modalities = marqo_index_model.properties.get('supported_modalities') or \
                                marqo_index_model.properties.get('supportedModalities') or []

    with metric_obj.time(f"{_id}.thread_time"):
        for doc in allocated_docs:
            for field in list(doc):
                if field not in tensor_fields:
                    continue
                if isinstance(doc[field], str) or force_download:
                    try:    
                        inferred_modality = infer_modality(doc[field], media_download_headers)
                    except MediaDownloadError as e:
                        if is_structured_index and media_field_types_mapping[field] == FieldType.ImagePointer:
                            # Continue processing for structured indexes with image fields
                            inferred_modality = Modality.IMAGE
                        else:
                            media_repo[doc[field]] = MediaDownloadError(f"Error inferring modality of media file {doc[field]}: {e}")
                            continue

                    if (inferred_modality == Modality.IMAGE and is_unstructured_index) or (
                        is_structured_index and media_field_types_mapping[field] == FieldType.ImagePointer): # Don't use infer modality in structured image pointers

                        if marqo_index_model.properties.get('type') in [ModelType.LanguageBind] \
                            and supported_modalities is not None \
                            and Modality.IMAGE not in supported_modalities:

                            media_repo[doc[field]] = UnsupportedModalityError(
                                f"Model {marqo_index_model.name} does not support {inferred_modality}")
                            continue

                        # Existing logic
                        if doc[field] in media_repo:
                            continue

                        try:
                            media_repo[doc[field]] = clip_utils.load_image_from_path(doc[field], media_download_headers,
                                                                                     timeout_ms=int(
                                                                                         TIMEOUT_SECONDS * 1000),
                                                                                     metrics_obj=metric_obj)
                        except PIL.UnidentifiedImageError as e:
                            media_repo[doc[field]] = e
                            metric_obj.increment_counter(f"{doc.get(field, '')}.UnidentifiedImageError")
                            continue
                        # preprocess image to tensor
                        if preprocessors is not None and preprocessors.image is not None:
                            if not device or not isinstance(device, str):
                                raise ValueError("Device must be provided for preprocessing images")
                            try:
                                preprocessed_results = preprocessors.image(media_repo[doc[field]])
                                if isinstance(preprocessed_results, torch.Tensor):
                                    media_repo[doc[field]] = preprocessed_results.to(device)
                                elif isinstance(preprocessed_results, dict):
                                    media_repo[doc[field]] = {k: v.to(device) for k, v in preprocessed_results.items()}
                                # media_repo[doc[field]] = preprocessors.image(media_repo[doc[field]]).to(device)
                            except OSError as e:
                                if "image file is truncated" in str(e):
                                    media_repo[doc[field]] = e
                                    metric_obj.increment_counter(f"{doc.get(field, '')}.OSError")
                                    continue
                                else:
                                    raise e

                    elif (inferred_modality in [Modality.VIDEO, Modality.AUDIO] and is_unstructured_index) or (
                            is_structured_index and media_field_types_mapping[field] in [FieldType.AudioPointer, FieldType.VideoPointer] and inferred_modality in [Modality.AUDIO, Modality.VIDEO]):
                        if marqo_index_model.properties.get('type') not in [ModelType.LanguageBind]:
                            media_repo[doc[field]] = UnsupportedModalityError(
                                f"Model {marqo_index_model.name} does not support {inferred_modality}")
                            continue
                        
                        if inferred_modality not in supported_modalities:
                            media_repo[doc[field]] = UnsupportedModalityError(
                                f"Model {marqo_index_model.name} does not support {inferred_modality}")
                            continue

                        if is_structured_index:
                            if inferred_modality is Modality.VIDEO and media_field_types_mapping[
                            field] is FieldType.AudioPointer:
                                media_repo[doc[field]] = MediaMismatchError(
                                    f"Invalid audio file. Error processing media file {doc}, detected as video, but field type is not VideoPointer")
                                continue

                            if inferred_modality is Modality.AUDIO and media_field_types_mapping[
                            field] is FieldType.VideoPointer:
                                media_repo[doc[field]] = MediaMismatchError(
                                    f"Invalid video file. Error processing media file {doc}, detected as audio, but field type is not AudioPointer")
                                continue

                        try:
                            processed_chunks = download_and_chunk_media(
                                url=doc[field], device=device, modality=inferred_modality,
                                preprocessors=preprocessors, audio_preprocessing=audio_preprocessing,
                                video_preprocessing=video_preprocessing, media_download_headers=media_download_headers
                            )
                            media_repo[doc[field]] = processed_chunks
                        except (ffmpeg.Error, S2InferenceError) as e:
                            logger.error(f"Error processing {inferred_modality} file: {str(e)}")
                            media_repo[doc[field]] = S2InferenceError(f"Error processing {inferred_modality} file: {str(e)}")
                    
                    elif inferred_modality is Modality.TEXT and is_structured_index and media_field_types_mapping[field] in [FieldType.AudioPointer, FieldType.VideoPointer, FieldType.ImagePointer]:
                        media_repo[doc[field]] = S2InferenceError(f"Error processing media file {doc}, detected as text, expected a {media_field_types_mapping[field]} pointer")
                    else:
                        pass
                else:
                    raise InternalError(f"Invalid field type for {field} to be added in media repo. Must be a string "
                                        f"but {type(field)}.")


def download_and_chunk_media(url: str, device: str, modality: Modality,
                             preprocessors: Preprocessors, audio_preprocessing: AudioPreProcessing = None,
                             video_preprocessing: VideoPreProcessing = None,
                             media_download_headers: Optional[Dict] = None) -> List[Dict[str, torch.Tensor]]:
    MAX_FILE_SIZE = read_env_vars_and_defaults_ints(EnvVars.MARQO_MAX_ADD_DOCS_VIDEO_AUDIO_FILE_SIZE)

    processor = StreamingMediaProcessor(
        url=url, device=device, modality=modality, preprocessors=preprocessors,
        audio_preprocessing=audio_preprocessing, video_preprocessing=video_preprocessing,
        media_download_headers=media_download_headers, enable_video_gpu_acceleration=_enable_video_gpu_acceleration()
    )

    if processor.total_size > MAX_FILE_SIZE:
        raise MediaExceedsMaxSizeError(
            f"File size ({processor.total_size / 1024 / 1024:.2f} MB) exceeds the maximum allowed size of {MAX_FILE_SIZE / 1024 / 1024:.2f} MB")

    return processor.process_media()


def _enable_video_gpu_acceleration() -> bool:
    """A helper function to determine if the video decoding should be done on the GPU.

    The environment variable MARQO_ENABLE_VIDEO_GPU_ACCELERATION is set on marqo start_on script.
    """
    return utils.read_env_vars_and_defaults(EnvVars.MARQO_ENABLE_VIDEO_GPU_ACCELERATION) == 'TRUE'


@contextmanager
def download_and_preprocess_multimedia_content(
        docs: List[Dict[str, str]],
        media_field_types_mapping: Dict[str, FieldType],
        marqo_index: MarqoIndex,
        add_docs_params: AddDocsParams
) -> ContextManager[dict]:
    thread_count = _determine_thread_count(marqo_index, add_docs_params)

    media_repo = process_batch(
        docs=docs,
        thread_count=thread_count,
        tensor_fields=list(media_field_types_mapping.keys()),
        media_field_types_mapping=media_field_types_mapping,
        media_download_headers=add_docs_params.media_download_headers,
        marqo_index_type=marqo_index.type,
        device=add_docs_params.device,
        marqo_index_model=marqo_index.model,
        model_name=marqo_index.model.name,
        model_properties=marqo_index.model.properties,
        normalize_embeddings=marqo_index.normalize_embeddings,
        model_auth=add_docs_params.model_auth,
        patch_method_exists=marqo_index.image_preprocessing.patch_method is not None,
        audio_preprocessing=marqo_index.audio_preprocessing,
        video_preprocessing=marqo_index.video_preprocessing,
        force_download=False,  # TODO verify if this is used
    )

    try:
        yield media_repo
    finally:
        for p in media_repo.values():
            if isinstance(p, ImageFile):
                p.close()
            elif isinstance(p, (list, np.ndarray)):
                # Clean up video/audio chunks if necessary
                pass


def _determine_thread_count(marqo_index: MarqoIndex, add_docs_params: AddDocsParams):
    # TODO this logic is copied from tensor search. Can be simplified and moved to AddDocsParams?
    model_properties = marqo_index.model.get_properties()
    is_languagebind_model = model_properties.get('type') == 'languagebind'

    default_image_thread_count = 20
    default_media_thread_count = 5

    # Check if media_download_thread_count is set in params
    if (add_docs_params.media_download_thread_count is not None and
            add_docs_params.media_download_thread_count != default_media_thread_count):
        return add_docs_params.media_download_thread_count

    env_media_thread_count = os.environ.get(EnvVars.MARQO_MEDIA_DOWNLOAD_THREAD_COUNT_PER_REQUEST)
    if env_media_thread_count is not None and int(env_media_thread_count) != default_media_thread_count:
        return int(env_media_thread_count)

    # If it's a LanguageBind model and no explicit setting, use 5
    if is_languagebind_model:
        return 5

    # Check if image_download_thread_count is explicitly set in params
    if (add_docs_params.image_download_thread_count is not None and
            add_docs_params.image_download_thread_count != default_image_thread_count):
        return add_docs_params.image_download_thread_count

    # Check if environment variable is explicitly set
    env_image_thread_count = os.environ.get(EnvVars.MARQO_IMAGE_DOWNLOAD_THREAD_COUNT_PER_REQUEST)
    if env_image_thread_count is not None and int(env_image_thread_count) != default_image_thread_count:
        return int(env_image_thread_count)

    # Default case
    return default_image_thread_count


@contextmanager
def download_and_preprocess_content(docs: List[dict], thread_count: int, tensor_fields: List[str],
                                    model_name: str,
                                    normalize_embeddings: bool,
                                    media_field_types_mapping: Optional[Dict[str, FieldType]],
                                    media_download_headers: Optional[Dict] = None,
                                    model_properties: Optional[Dict] = None,
                                    model_auth: Optional[ModelAuth] = None,
                                    device: Optional[str] = None,
                                    patch_method_exists: bool = False,
                                    marqo_index_type: Optional[IndexType] = None,
                                    marqo_index_model: Optional[Model] = None,
                                    audio_preprocessing: Optional[AudioPreProcessing] = None,
                                    video_preprocessing: Optional[VideoPreProcessing] = None,
                                    force_download: bool = False
                                    ) -> ContextManager[dict]:
    media_repo = {}  # for image/video/audio
    media_repo = process_batch(
        docs = docs,
        thread_count = thread_count,
        tensor_fields = tensor_fields,
        model_name = model_name,
        normalize_embeddings = normalize_embeddings,
        force_download = force_download,
        media_field_types_mapping = media_field_types_mapping,
        media_download_headers = media_download_headers,
        model_properties = model_properties,
        model_auth = model_auth,
        device = device,
        patch_method_exists = patch_method_exists,
        marqo_index_type = marqo_index_type,
        marqo_index_model = marqo_index_model,
        audio_preprocessing = audio_preprocessing,
        video_preprocessing = video_preprocessing
    )

    try:
        yield media_repo
    finally:
        for p in media_repo.values():
            if isinstance(p, ImageFile):
                p.close()
            elif isinstance(p, (list, np.ndarray)):
                # Clean up video/audio chunks if necessary
                pass


def process_batch(
        docs: List[dict], thread_count: int, tensor_fields: List[str],
        model_name: str, normalize_embeddings: bool,
        force_download: bool, media_field_types_mapping: Optional[Dict[str, FieldType]],
        model_properties: Optional[Dict],
        model_auth: Optional[ModelAuth], device: Optional[str],
        patch_method_exists: bool, marqo_index_type: Optional[IndexType], marqo_index_model: Optional[Model],
        media_download_headers: Optional[Dict] = None,
        audio_preprocessing: Optional[AudioPreProcessing] = None,
        video_preprocessing: Optional[VideoPreProcessing] = None
) -> dict:

    docs_per_thread = math.ceil(len(docs) / thread_count)
    copied = copy.deepcopy(docs)

    if is_preprocessor_preload(model_properties) and not patch_method_exists:
        _, preprocessors = load_multimodal_model_and_get_preprocessors(
            model_name=model_name,
            model_properties=model_properties,
            device=device,
            model_auth=model_auth,
            normalize_embeddings=normalize_embeddings
        )
    else:
        preprocessors = None

    media_repo = {}
    m = [RequestMetrics() for i in range(thread_count)]
    # Consider replacing below with:
    # thread_allocated_docs = [copied[i: i + docs_per_thread] for i in range(0, len(copied), docs_per_thread)]
    thread_allocated_docs = [copied[i: i + docs_per_thread] for i in range(len(copied))[::docs_per_thread]]

    with ThreadPoolExecutor(max_workers=len(thread_allocated_docs)) as executor:
        futures = [
            executor.submit(
                threaded_download_and_preprocess_content,
                allocation,
                media_repo,
                tensor_fields,
                device,
                media_field_types_mapping,
                media_download_headers,
                m[i],
                preprocessors,
                marqo_index_type,
                marqo_index_model,
                audio_preprocessing,
                video_preprocessing,
                force_download)
            for i, allocation in enumerate(thread_allocated_docs)
        ]

        # Unhandled exceptions will be raised here.
        # We only raise the first exception if there are multiple exceptions
        for future in concurrent.futures.as_completed(futures):
            _ = future.result()

    # Fix up metric_obj to make it not mention thread-ids
    metric_obj = RequestMetricsStore.for_request()
    metric_obj = RequestMetrics.reduce_from_list([metric_obj] + m)
    metric_obj.times = reduce_thread_metrics(metric_obj.times)
    return media_repo


def reduce_thread_metrics(data):
    """Reduce the metrics from each thread, as if they were run in a single thread.

    e.g.
    ```
    {
        "image_download.700.thread_time": 1373.271582997404,
        "image_download.700.https://www.ai-nc.com/images/pages/heat-map.png": 52.985392,
        "image_download.729.thread_time": 53.297404,
        "image_download.729.https://www.ai-nc.com/images/pages/heat-map.png": 2052.617332985392,
    }
    ```
    Becomes
    ```
    {
        "image_download.thread_time": [1373.271582997404, 53.297404],
        "image_download.https://www.ai-nc.com/images/pages/heat-map.png": [2052.617332985392, 52.985392],
    }
    ```
    Only applies to times that start with `image_download`.
    """
    result = {}
    for key, value in data.items():
        if key.startswith("image_download."):
            parts = key.split('.')
            new_key = '.'.join(parts[0:1] + parts[2:]) if parts[1] != 'full_time' else key
            if new_key in result:
                if isinstance(result[new_key], list):
                    result[new_key].append(value)
                else:
                    result[new_key] = [result[new_key], value]
            else:
                result[new_key] = value
    return result


def determine_document_dict_field_type(field_name: str, field_content, mappings: dict) -> FieldType:
    """
    Only used for unstructured. Structured indexes have field types declared upon index creation.
    Determines the type of a document field if it is a dict
    using its name, content, and the add docs mappings object.
    3 Options:
    1. `None` if standard (str, int, float, bool, list)
    2. `MultimodalCombination` (dict)
    3. `CustomVector` (dict)
    4. Add other dict types as needed
    """

    if isinstance(field_content, dict):
        if field_name not in mappings:
            raise base_exceptions.InternalError(
                f"Invalid dict field {field_name}. Could not find field in mappings object.")

        if mappings[field_name]["type"] == enums.MappingsObjectType.multimodal_combination:
            return enums.MappingsObjectType.multimodal_combination
        elif mappings[field_name]["type"] == enums.MappingsObjectType.custom_vector:
            return enums.MappingsObjectType.custom_vector
        else:
            raise base_exceptions.InternalError(
                f"Invalid dict field type: '{mappings[field_name]['type']}' for field: '{field_name}' in mappings. Must be one of {[t.value for t in enums.MappingsObjectType]}")
    else:
        return None
