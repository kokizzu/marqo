import asyncio
import io
import os
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError
from typing import Dict, Any, List, Optional, Union, Tuple
from urllib.parse import urlparse

import httpcore
import httpx
import orjson

import marqo.logging
import marqo.vespa.concurrency as conc
from marqo.core.models import MarqoIndex
from marqo.core.semi_structured_vespa_index.common import VESPA_DOC_FIELD_TYPES, VESPA_DOC_CREATE_TIMESTAMP
from marqo.core.semi_structured_vespa_index.marqo_field_types import MarqoFieldTypes
from marqo.marqo_docs import update_documents_response
from marqo.vespa.exceptions import (VespaStatusError, VespaError, InvalidVespaApplicationError,
                                    VespaTimeoutError, VespaNotConvergedError, VespaActivationConflictError)
from marqo.vespa.models import VespaDocument, QueryResult, Error, FeedBatchDocumentResponse, FeedBatchResponse, \
    FeedDocumentResponse, UpdateDocumentsBatchResponse, UpdateDocumentResponse, FeedBatchDocumentResponse
from marqo.vespa.models.application_metrics import ApplicationMetrics
from marqo.vespa.models.delete_document_response import DeleteDocumentResponse, DeleteBatchDocumentResponse, \
    DeleteBatchResponse, DeleteAllDocumentsResponse
from marqo.vespa.models.get_document_response import GetDocumentResponse, VisitDocumentsResponse, GetBatchResponse, \
    GetBatchDocumentResponse

logger = marqo.logging.get_logger(__name__)


class VespaClient:
    _VESPA_ERROR_CODE_TO_EXCEPTION = {
        'INVALID_APPLICATION_PACKAGE': InvalidVespaApplicationError,
        'ACTIVATION_CONFLICT': VespaActivationConflictError
    }

    class _ConvergenceStatus:
        def __init__(self, current_generation: int, wanted_generation: int, converged: bool):
            self.current_generation = current_generation
            self.wanted_generation = wanted_generation
            self.converged = converged

    def __init__(self, config_url: str, document_url: str, query_url: str,
                 content_cluster_name: str, default_search_timeout_ms: int = 1000,
                 pool_size: int = 10, feed_pool_size: int = 10, get_pool_size: int = 10,
                 delete_pool_size: int = 10, partial_update_pool_size: int = 10):
        """
        Create a VespaClient object.
        Args:
            config_url: Vespa Deploy API base URL
            document_url: Vespa Document API base URL
            query_url: Vespa Query API base URL
            pool_size: Number of connections to keep in the connection pool
            feed_pool_size: Number of connections to keep in batch feed requests connection pool to Vespa
            get_pool_size: Number of connections to keep in batch get requests connection pool to Vespa
            delete_pool_size: Number of connections to keep batch delete requests connection pool to Vespa
            partial_update_pool_size: Number of connections to keep batch partial update requests connection pool to Vespa
        """
        self.config_url = config_url.strip('/')
        self.document_url = document_url.strip('/')
        self.query_url = query_url.strip('/')
        self.http_client = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=pool_size, max_connections=pool_size)
        )
        self.default_search_timeout_ms = default_search_timeout_ms
        self.content_cluster_name = content_cluster_name
        self.feed_pool_size = feed_pool_size
        self.get_pool_size = get_pool_size
        self.delete_pool_size = delete_pool_size
        self.partial_pool_size = partial_update_pool_size

    def close(self):
        """
        Close the VespaClient object.
        """
        self.http_client.close()

    def deploy_application(self, application: str, timeout: int = 60) -> None:
        """
        Deploy a Vespa application.
        Args:
            application: Path to the Vespa application root directory
            timeout: Timeout in seconds
        """
        endpoint = f'{self.config_url}/application/v2/tenant/default/prepareandactivate'

        gzip_stream = self._gzip_compress(application)

        response = self.http_client.post(
            endpoint,
            headers={'Content-Type': 'application/x-gzip'},
            data=gzip_stream.read(),
            timeout=timeout
        )

        self._raise_for_status(response)

    def create_deployment_session(self, check_for_application_convergence: bool = True) -> Tuple[str, str]:
        """
        Create a Vespa deployment session.
        Args:
            check_for_application_convergence: check for the application to converge before create a deployment session.

        Returns:
            Tuple[str, str]:
             - content_base_url is the base url for contents in this session
             - prepare_url is the url for prepare this session

        Please note that the session created is local in one config server and will be replicated to multiple servers
        via Zookeeper. Following requests should use content_base_url and prepare_url to make sure it can hit the right
        config server that this session is created on.
        """
        if check_for_application_convergence:
            self.check_for_application_convergence()

        res = self._create_deploy_session(self.http_client)
        content_base_url = res['content']
        prepare_url = res['prepared']
        return content_base_url, prepare_url

    def download_application(self, check_for_application_convergence: bool = False) -> str:
        """
        Args:
            check_for_application_convergence: check for the application to converge before downloading.

        Download the Vespa application. If wait_for_application_convergence is True, this method will wait for the
        application to converge before downloading.

        Application download happens in two steps:
        1. Create a session
        2. Download the application using the session ID

        The session created in step 1 is local to the config node that created it and subsequent requests will return a
        404 error if the request is routed to a different config node. This method attempts to ensure the same config
        node is used for all requests by using the same httpx client for all requests. However, this is not guaranteed.

        The likelihood of getting a 404 error is further reduced if config cluster uses a load balancer with sticky
        sessions. Since we are using a single httpx client, cookie-based sticky sessions will work with this
        implementation.

        Returns:
            Path to the downloaded application
        """
        if check_for_application_convergence:
            self.check_for_application_convergence()

        with httpx.Client() as httpx_client:
            session_id = self._create_deploy_session(httpx_client)['session-id']
            return self._download_application(session_id, httpx_client)

    def check_for_application_convergence(self) -> None:
        """
        Check if the Vespa application has converged and raise an exception if it has not.

        Raises:
            VespaNotConvergedError: If the application has not converged
        """
        if not self.get_application_has_converged():
            raise VespaNotConvergedError('Vespa application has not converged')

    def get_application_generation(self) -> int:
        """
        Get the current application generation.

        Returns:
            Current application generation
        """
        return self._get_convergence_status().current_generation

    def get_application_has_converged(self) -> bool:
        """
        Get the current application convergence status.

        Application convergence is asynchronous following a deployment. This method can be used to check if the
        application has converged after a deployment.

        Returns:
            True if the application is converged, False otherwise
        """
        return self._get_convergence_status().converged

    def wait_for_application_convergence(self, timeout: int = 120) -> None:
        """
        Wait for Vespa application to converge, checking every second.

        Args:
            timeout: Timeout in seconds
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if self.get_application_has_converged():
                    return
                else:
                    logger.debug('Waiting for Vespa application to converge')
                    time.sleep(1)
            # TODO Find out what exceptions is raised here
            except (httpx.TimeoutException, httpcore.TimeoutException):
                logger.error("Marqo timed out waiting for Vespa application to converge. Will retry.")

        raise VespaError(f"Vespa application did not converge within {timeout} seconds. "
                         f"The convergence status is {self._get_convergence_status()}")

    def query(self, yql: str, hits: int = 10, ranking: str = None, model_restrict: str = None,
              query_features: Dict[str, Any] = None, timeout: float = None, **kwargs) -> QueryResult:
        """
        Query Vespa.
        Args:
            yql: YQL query
            hits: Number of hits to return
            ranking: Ranking profile to use
            model_restrict: Schema to restrict the query to
            query_features: Query features
            **kwargs: Additional query parameters
        Returns:
            Query result as a VespaQueryResult object
        """
        query_features_list = {
            f'input.query({key})': value for key, value in query_features.items()
        } if query_features else {}

        query = {
            'yql': yql,
            'hits': hits,
            'ranking': ranking,
            'model.restrict': model_restrict,
            **query_features_list,
            **kwargs
        }

        # Use default timeout if not already set.
        if timeout:
            query['timeout'] = f"{timeout}ms"
        else:
            query['timeout'] = f"{self.default_search_timeout_ms}ms"

        query = {key: value for key, value in query.items() if value is not None}

        logger.debug(f'Query: {query}')

        try:
            resp = self.http_client.post(f'{self.query_url}/search/', json=query)
        except httpx.HTTPError as e:
            raise VespaError(e) from e

        self._query_raise_for_status(resp)

        return QueryResult(**orjson.loads(resp.text))

    def feed_document(self, document: VespaDocument, schema: str, timeout: int = 60) -> FeedDocumentResponse:
        """
        Feed a document to Vespa.

        Args:
            document: Document to feed
            schema: Schema to feed to
            timeout: Timeout in seconds

        Returns:
            FeedResponse object
        """
        doc_id = document.id
        data = {'fields': document.fields}

        end_point = f'{self.document_url}/document/v1/{schema}/{schema}/docid/{doc_id}'

        resp = self.http_client.post(end_point, json=data, timeout=timeout)

        self._raise_for_status(resp)

        return FeedDocumentResponse(**resp.json())

    def feed_batch(self,
                   batch: List[VespaDocument],
                   schema: str,
                   concurrency: Optional[int] = None,
                   timeout: int = 60) -> FeedBatchResponse:
        """
        Feed a batch of documents to Vespa concurrently.

        Documents will be fed with `concurrency` concurrent pooled connections.

        Args:
            batch: List of documents to feed
            schema: Schema to feed to
            concurrency: Number of concurrent feed requests
            timeout: Timeout in seconds per request

        Returns:
            A FeedBatchResponse object
        """
        if not batch:
            return FeedBatchResponse(responses=[], errors=False)

        if concurrency is None:
            concurrency = self.feed_pool_size

        batch_response = conc.run_coroutine(
            self._feed_batch_async(batch, schema, concurrency, timeout)
        )

        return batch_response

    def feed_batch_sync(self, batch: List[VespaDocument], schema: str) -> FeedBatchResponse:
        """
        Feed a batch of documents to Vespa sequentially.

        This method is for debugging and experimental purposes only. Sequential feeding can be very slow.

        Args:
            batch: List of documents to feed
            schema: Schema to feed to

        Returns:
            List of FeedResponse objects
        """
        responses = [
            self._feed_document_sync(self.http_client, document, schema, timeout=60)
            for document in batch
        ]

        errors = False
        for response in responses:
            if response.status != 200:
                errors = True

        return FeedBatchResponse(responses=responses, errors=errors)

    def feed_batch_multithreaded(self, batch: List[VespaDocument], schema: str,
                                 max_threads: int = 10) -> FeedBatchResponse:
        """
        Feed a batch of documents to Vespa concurrently using a thread pool.

        This method is for debugging and experimental purposes only. Use `feed_batch` instead to feed documents
        asynchronously with one thread.

        Args:
            batch: List of documents to feed
            schema: Schema to feed to
            max_threads: Maximum number of threads to use

        Returns:
            List of FeedResponse objects
        """
        with httpx.Client(
                limits=httpx.Limits(max_keepalive_connections=max_threads, max_connections=max_threads)) as sync_client:
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                responses = list(executor.map(
                    lambda document: self._feed_document_sync(sync_client, document, schema, timeout=60), batch
                ))

        errors = False
        for response in responses:
            if response.status != 200:
                errors = True

        return FeedBatchResponse(responses=responses, errors=errors)

    def get_document(self, id: str, schema: str) -> GetDocumentResponse:
        """
        Get a document by ID.

        Args:
            id: Document ID
            schema: Schema to get from

        Returns:
            GetDocumentResponse object
        """
        try:
            resp = self.http_client.get(f'{self.document_url}/document/v1/{schema}/{schema}/docid/{id}')
        except httpx.HTTPError as e:
            raise VespaError(e) from e

        self._raise_for_status(resp)

        return GetDocumentResponse(**resp.json())

    def get_all_documents(self,
                          schema: str,
                          stream=False,
                          continuation: Optional[str] = None
                          ) -> VisitDocumentsResponse:
        """
        Get all documents in a schema.
        Args:
            schema: Schema to get from
            stream: Whether to stream the response
            continuation: Continuation token for pagination

        Returns:
            BatchGetDocumentResponse object
        """
        try:
            url = self._add_query_params(
                url=f'{self.document_url}/document/v1/{schema}/{schema}/docid',
                query_params={
                    'stream': str(stream).lower(),
                    'continuation': continuation
                }
            )
            logger.debug(f'URL: {url}')
            resp = self.http_client.get(url)
        except httpx.HTTPError as e:
            raise VespaError(e) from e

        self._raise_for_status(resp)

        return VisitDocumentsResponse(**resp.json())

    def get_batch(self,
                  ids: List[str],
                  schema: str,
                  fields: Optional[List[str]] = None,
                  concurrency: Optional[int] = None,
                  timeout: int = 60) -> GetBatchResponse:
        """
        Get a batch of documents by ID concurrently.

        Documents will be fetched with `concurrency` concurrent pooled connections.

        Missing (404) documents will be returned in the response. Any other non-200 responses will raise an exception.

        Args:
            ids: List of document IDs to get
            schema: Schema to get from
            fields: A optional list of fields to fetch from the document
            concurrency: Number of concurrent get requests
            timeout: Timeout in seconds per request

        Returns:
            List of GetDocumentResponse objects containing the documents fetched and any missing documents (404)
        """
        if not ids:
            return GetBatchResponse(responses=[], errors=False)

        if concurrency is None:
            concurrency = self.get_pool_size

        batch_response = conc.run_coroutine(
            self._get_batch_async(ids, fields, schema, concurrency, timeout)
        )

        return batch_response

    def delete_document(self, id: str, schema: str) -> DeleteDocumentResponse:
        """
        Delete a document by ID.

        Note that this method returns a successful response even if the document does not exist.

        Args:
            id: Document ID
            schema: Schema to delete from
        """
        try:
            resp = self.http_client.delete(f'{self.document_url}/document/v1/{schema}/{schema}/docid/{id}')
        except httpx.HTTPError as e:
            raise VespaError(e) from e

        self._raise_for_status(resp)

        return DeleteDocumentResponse(**resp.json())

    def delete_all_docs(self, schema: str) -> DeleteAllDocumentsResponse:
        """Deletes all documents in the given index"""
        try:
            resp = self.http_client.delete(f'{self.document_url}/document/v1/{schema}'
                                           f'/{schema}/docid/?cluster={self.content_cluster_name}&selection=true')
        except httpx.HTTPError as e:
            raise VespaError(e) from e

        self._raise_for_status(resp)
        return DeleteAllDocumentsResponse(**resp.json())

    def delete_batch(self,
                     ids: List[str],
                     schema: str,
                     concurrency: Optional[int] = None,
                     timeout: int = 60) -> DeleteBatchResponse:
        """
        Delete a batch of documents by ID concurrently.

        Documents will be deleted with `concurrency` concurrent pooled connections.

        Args:
            ids: List of document IDs to delete
            schema: Schema to delete from
            concurrency: Number of concurrent delete requests
            timeout: Timeout in seconds per request

        Returns:
            A DeleteBatchResponse object
        """
        if not ids:
            return DeleteBatchResponse(responses=[], errors=False)

        if concurrency is None:
            concurrency = self.delete_pool_size

        batch_response = conc.run_coroutine(
            self._delete_batch_async(ids, schema, concurrency, timeout)
        )

        return batch_response

    def update_documents_batch(self, batch: List[VespaDocument],
                               schema: str,
                               concurrency: Optional[int] = None,
                               timeout: int = 60,
                               vespa_id_field: str = "marqo__id") -> UpdateDocumentsBatchResponse:
        """
        Partial update documents in batch concurrently.

        If the document does not exist, it will not be created and an error for that document will be
        returned in the response.

        Args:
            batch: A list of documents to update
            schema: schema name
            concurrency: Number of concurrent delete requests, can be configured by environment variable
                VESPA_PARTIAL_UPDATE_POOL_SIZE
            timeout: Timeout in seconds per request
            vespa_id_field: The field name of the vespa document id under the fields dictionary

        Returns:
            A UpdateDocumentsBatchResponse object
        """

        if not batch:
            return UpdateDocumentsBatchResponse(responses=[], errors=False)

        if concurrency is None:
            concurrency = self.partial_pool_size

        batch_response = conc.run_coroutine(
            self._update_documents_batch_async(batch, schema, concurrency, timeout, vespa_id_field)
        )

        return batch_response

    def get_metrics(self) -> ApplicationMetrics:
        """
        Get metrics for every service on all nodes for the application.

        See https://docs.vespa.ai/en/operations-selfhosted/monitoring.html#metrics-v2-values for more information.

        Returns:
             A selected set of metrics for every service on all nodes for the application
        """
        try:
            resp = self.http_client.get(f'{self.document_url}/metrics/v2/values')
        except httpx.HTTPError as e:
            raise VespaError(e) from e

        self._raise_for_status(resp)

        return ApplicationMetrics(**resp.json())

    def get_index_setting_by_name(self, index_name: str) -> Optional[MarqoIndex]:
        try:
            resp = self.http_client.get(f'{self.document_url}/index-settings/{index_name}')
        except httpx.HTTPError as e:
            raise VespaError(e) from e

        if resp.status_code == 404:
            return None

        self._raise_for_status(resp)

        return MarqoIndex.parse_obj(resp.json())

    def get_all_index_settings(self) -> List[MarqoIndex]:
        try:
            resp = self.http_client.get(f'{self.document_url}/index-settings')
        except httpx.HTTPError as e:
            raise VespaError(e) from e

        self._raise_for_status(resp)

        index_list = resp.json()
        if isinstance(index_list, list):
            return [MarqoIndex.parse_obj(item) for item in index_list]

        raise VespaError(f'Get all index settings returns invalid response: {index_list}')

    def translate_vespa_document_response(self, status: int, message: Optional[str]=None) -> Tuple[int, Optional[str]]:
        """A helper function to translate Vespa document response into the expected status, message that
        is used in Marqo document API responses.

        Args:
            status: The status code from Vespa document response

        Return:
            A tuple of status code and the message in the response
        """
        vespa_status_code_to_marqo_doc_error_map = {
            200: (200, None),
            404: (404, "Document does not exist in the index"),
            412: (400, "Marqo vector store couldn't update the document. Please see: " + update_documents_response() + " for more details"), # Update documents get 412 from Vespa for document not found as we use condition
            429: (429, "Marqo vector store receives too many requests. Please try again later"),
            507: (400, "Marqo vector store is out of memory or disk space"),
        }

        if status in vespa_status_code_to_marqo_doc_error_map:
            return vespa_status_code_to_marqo_doc_error_map[status]
        elif status == 400 and isinstance(message, str) and "could not parse field" in message.lower():
            # TODO Block the invalid special characters before sending to Vespa
            return 400, f"The document contains invalid characters in the fields. Original error: {message} "
        else:
            logger.error(f"An unexpected error occurred from the Vespa document response. "
                         f"status: {status}, message: {message}")
            return 500, f"Marqo vector store returns an unexpected error with this document. Original error: {message}"

    def _add_query_params(self, url: str, query_params: Dict[str, str]) -> str:
        if not query_params:
            return url

        query_string = '&'.join([f'{key}={value}' for key, value in query_params.items() if value])
        return f'{url.strip("?")}?{query_string}'

    def _gzip_compress(self, directory: str) -> io.BytesIO:
        """
        Gzip all files in the given directory and return an in-memory byte buffer.
        """
        byte_stream = io.BytesIO()
        with tarfile.open(fileobj=byte_stream, mode='w:gz') as tar:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, directory)  # archive name should be relative
                    tar.add(file_path, arcname=arcname)

        byte_stream.seek(0)
        return byte_stream

    def _create_deploy_session(self, httpx_client: httpx.Client) -> Dict:
        endpoint = f'{self.config_url}/application/v2/tenant/default/session?from=' \
                   f'{self.config_url}/application/v2/tenant/default/application/default/environment' \
                   f'/default/region/default/instance/default'

        response = httpx_client.post(endpoint)

        self._raise_for_status(response)

        return response.json()

    def get_content_url(self, content_base_url: str, *paths: str) -> str:
        return f'{content_base_url}{"/".join(paths)}'

    def list_contents(self, content_base_url: str) -> List[str]:
        endpoint = f'{content_base_url}?recursive=true'

        response = self.http_client.get(endpoint)

        self._raise_for_status(response)

        return response.json()

    def get_text_content(self, content_base_url: str, *path: str) -> str:
        endpoint = f'{content_base_url}{"/".join(path)}'

        response = self.http_client.get(endpoint)

        self._raise_for_status(response)

        return response.text

    def get_binary_content(self, content_base_url: str, *path: str) -> bytes:
        endpoint = f'{content_base_url}{"/".join(path)}'

        response = self.http_client.get(endpoint)

        self._raise_for_status(response)

        return response.content

    def put_content(self, content_base_url: str, content: Union[str, bytes], *path: str) -> None:
        endpoint = f'{content_base_url}{"/".join(path)}'

        response = self.http_client.put(endpoint, content=content)

        self._raise_for_status(response)

    def delete_content(self, content_base_url: str, *path: str) -> None:
        endpoint = f'{content_base_url}{"/".join(path)}'

        response = self.http_client.delete(endpoint)

        self._raise_for_status(response)

    def prepare(self, prepare_url: str, timeout: int):
        response = self.http_client.put(prepare_url, timeout=timeout)

        self._raise_for_status(response)

        return response.json()

    def activate(self, activate_url: str, timeout: int):
        response = self.http_client.put(activate_url, timeout=timeout)

        self._raise_for_status(response)

        return response.json()

    def get_vespa_version(self) -> str:
        endpoint = f'{self.config_url}/state/v1/version'

        response = self.http_client.get(endpoint)

        self._raise_for_status(response)

        return response.json()['version']

    def _download_application(self, session_id: int, httpx_client: httpx.Client) -> str:
        endpoint = f'{self.config_url}/application/v2/tenant/default/session/{session_id}/content/?recursive=true'

        response = httpx_client.get(endpoint)

        self._raise_for_status(response)

        urls = response.json()

        logger.debug(f'URLs: {urls}')

        def is_file(url: str) -> bool:
            last_component = urlparse(url).path.split('/')[-1]
            return '.' in last_component

        temp_dir = tempfile.mkdtemp()

        logger.debug(f'Downloading application to {temp_dir}')

        for url in urls:
            if not is_file(url):
                continue  # Skip directories

            # Parse the URL
            parsed = urlparse(url)
            path_parts = parsed.path.split('/')

            # Find the index for 'content' and use it as root
            content_index = path_parts.index('content')
            rel_path = os.path.join(*path_parts[content_index + 1:])
            abs_path = os.path.join(temp_dir, rel_path)

            # Ensure directory exists before downloading
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

            response = httpx_client.get(url)
            self._raise_for_status(response)

            # Save the downloaded content
            with open(abs_path, 'wb') as f:
                f.write(response.content)

        return temp_dir

    def _get_convergence_status(self):
        endpoint = f'{self.config_url}/application/v2/tenant/default/application/default/environment/default/region/' \
                   f'default/instance/default/serviceconverge'

        response = self.http_client.get(endpoint)

        self._raise_for_status(response)

        try:
            json = response.json()
            return self._ConvergenceStatus(
                current_generation=json['currentGeneration'],
                wanted_generation=json['wantedGeneration'],
                converged=json['converged']
            )

        except (JSONDecodeError, KeyError) as e:
            raise VespaError(f'Unexpected response: {response.text}') from e

    async def _feed_batch_async(self, batch: List[VespaDocument],
                                schema: str,
                                connections: int, timeout: int) -> FeedBatchResponse:
        async with httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=connections,
                                                         max_connections=connections)) as async_client:
            semaphore = asyncio.Semaphore(connections)
            tasks = [
                asyncio.create_task(
                    self._feed_document_async(semaphore, async_client, document, schema, timeout)
                )
                for document in batch
            ]
            await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

        responses = []
        errors = False
        for task in tasks:
            result = task.result()
            responses.append(result)
            if result.status != 200:
                errors = True

        return FeedBatchResponse(responses=responses, errors=errors)

    async def _update_documents_batch_async(self, batch: List[VespaDocument],
                                            schema: str,
                                            connections: int, timeout: int,
                                            vespa_id_field: str) -> UpdateDocumentsBatchResponse:
        async with httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=connections,
                                                         max_connections=connections)) as async_client:
            semaphore = asyncio.Semaphore(connections)
            tasks = [
                asyncio.create_task(
                    self._update_document_async(semaphore, async_client, document, schema, timeout, vespa_id_field)
                )
                for document in batch
            ]
            await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

        responses = []
        errors = False
        for task in tasks:
            result = task.result()
            responses.append(result)
            if result.status != 200:
                errors = True

        return UpdateDocumentsBatchResponse(responses=responses, errors=errors)

    async def _update_document_async(self, semaphore: asyncio.Semaphore, async_client: httpx.AsyncClient,
                                     document: VespaDocument, schema: str,
                                     timeout: int, vespa_id_field: str) -> UpdateDocumentResponse:
        doc_id = document.id
        data = {'fields': document.fields}
        types = document.field_types
        create_timestamp = document.create_timestamp

        # only used for documents that are not updated
        error_doc_path_id = f"/document/v1/{schema}/{schema}/docid/{doc_id}"
        async with semaphore:
            end_point = f'{self.document_url}/document/v1/{schema}/{schema}/docid/{doc_id}?create=false'
            data["condition"] = f'{schema}.{vespa_id_field}==\"{doc_id}\"'
            if types is not None: # Types will be none for structured index as we are not storing types at the time of Add docs.
                for key, value in types.items():
                    data["condition"] += (f' and (not {schema}.{VESPA_DOC_FIELD_TYPES}{{\"{key}\"}} or {schema}.{VESPA_DOC_FIELD_TYPES}{{\"{key}\"}}==\"{value}\")'
                                          f' and (not ({schema}.{VESPA_DOC_FIELD_TYPES}{{\"{key}\"}}=="{MarqoFieldTypes.TENSOR.value}"))')
            if create_timestamp is not None:
                data["condition"] += f' and {schema}.{VESPA_DOC_CREATE_TIMESTAMP}=={create_timestamp}'
            try:
                resp = await async_client.put(end_point, json=data, timeout=timeout)
                if resp.status_code == 412 and types is None and create_timestamp is None:
                    # If Vespa response is 412, and the request is for structured index, it means the document does not exist
                    # in the index, as we don't have type checks / timestamp (version) checks for structured indexes.
                    # We return a 404 error for this case.
                    resp.status_code = 404
            except httpx.RequestError as e:
                logger.error(e, exc_info=True)
                return UpdateDocumentResponse(status=500, message="Network Error", id=doc_id, path_id=error_doc_path_id)

        # Handle other exceptions
        try:
            return UpdateDocumentResponse(**resp.json(), status=resp.status_code)
        except JSONDecodeError as e:
            if resp.status_code == 200:
                # A 200 response shouldn't reach here, so we error out the whole batch
                raise VespaError(cause=e, message=f"Unexpected response from Vespa: {resp.text}") from e

            try:
                self._raise_for_status(resp)
            except VespaStatusError as e:
                logger.error(e, exc_info=True)
                return UpdateDocumentResponse(status=resp.status_code, message=e.message, id=doc_id,
                                              error_doc_path_id=error_doc_path_id)

    async def _feed_document_async(self, semaphore: asyncio.Semaphore, async_client: httpx.AsyncClient,
                                   document: VespaDocument, schema: str,
                                   timeout: int) -> FeedBatchDocumentResponse:
        """An async method to feed a document to Vespa.

        Note: This method is used by the async feed batch method to feed documents concurrently. Unhandled exceptions
        will be raised in the main thread and leads a 500 error for the whole batch. Therefore, exceptions should be
        handled gracefully in this method for the specific document. We should keep the error message as similar as the
        Vespa error messages since this is a low level method. Overwrite the error message in higher level methods.

        Exceptions that are handled in this method:
        1. httpx.RequestError: We convert this error to a 500 error for the specific document and put 'Network Error' in
        the message.
        2. JSONDecodeError: If the Vespa response is 200 but the response can not be decoded, we raise a VespaError and
        this will block the whole batch as this indicates an unexpected response from Vespa.
        3. httpx.status_codes.HTTPStatusError: We catch the error and return it to marqo.core.document methods to handle
        it.

        Raises:
            VespaError: If the Vespa response is 200 but the response can not be decoded.

        Returns:
            FeedDocumentResponse object
        """
        doc_id = document.id
        data = {'fields': document.fields}

        async with semaphore:
            end_point = f'{self.document_url}/document/v1/{schema}/{schema}/docid/{doc_id}'
            # Handle httpx.RequestError
            try:
                resp = await async_client.post(end_point, json=data, timeout=timeout)
            except httpx.RequestError as e:
                logger.error(e, exc_info=True)
                return FeedBatchDocumentResponse(status=500, message="Network Error", id=doc_id)

        # Handle other exceptions
        try:
            return FeedBatchDocumentResponse(**resp.json(), status=resp.status_code)
        except JSONDecodeError as e:
            if resp.status_code == 200:
                # A 200 response shouldn't reach here, so we error out the whole batch
                raise VespaError(cause=e, message=f"Unexpected response from Vespa: {resp.text}") from e

            try:
                self._raise_for_status(resp)
            except VespaStatusError as e:
                logger.error(e, exc_info=True)
                return FeedBatchDocumentResponse(status=resp.status_code, message=e.message, id=doc_id)

    def _feed_document_sync(self, sync_client: httpx.Client, document: VespaDocument, schema: str,
                            timeout: int) -> FeedBatchDocumentResponse:
        doc_id = document.id
        data = {'fields': document.fields}

        end_point = f'{self.document_url}/document/v1/{schema}/{schema}/docid/{doc_id}'

        resp = sync_client.post(end_point, json=data, timeout=timeout)

        return FeedBatchDocumentResponse(**resp.json(), status=resp.status_code)

    async def _get_batch_async(self,
                               ids: List[str],
                               fields: Optional[List[str]],
                               schema: str,
                               connections: int, timeout: int) -> GetBatchResponse:
        async with httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=connections,
                                                         max_connections=connections)) as async_client:
            semaphore = asyncio.Semaphore(connections)
            tasks = [
                asyncio.create_task(
                    self._get_document_async(semaphore, async_client, id, fields, schema, timeout)
                )
                for id in ids
            ]
            await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

        responses = []
        errors = False
        for task in tasks:
            result = task.result()
            responses.append(result)
            if result.status != 200:
                errors = True

        return GetBatchResponse(responses=responses, errors=errors)

    async def _get_document_async(self,
                                  semaphore: asyncio.Semaphore,
                                  async_client: httpx.AsyncClient,
                                  id: str,
                                  fields: Optional[List[str]],
                                  schema: str,
                                  timeout: int) -> GetBatchDocumentResponse:
        async with semaphore:
            try:
                if fields is not None:
                    resp = await async_client.get(
                        f'{self.document_url}/document/v1/{schema}/{schema}/docid/{id}?fieldSet={schema}:{",".join(fields)}',
                        timeout=timeout
                    )
                else:
                    resp = await async_client.get(
                        f'{self.document_url}/document/v1/{schema}/{schema}/docid/{id}', timeout=timeout
                    )
            except httpx.HTTPError as e:
                raise VespaError(e) from e

            if resp.status_code in [200, 404]:
                return GetBatchDocumentResponse(**resp.json(), status=resp.status_code)

            self._raise_for_status(resp)

    async def _get_document_async_with_specific_fields(self,
                                  semaphore: asyncio.Semaphore,
                                  async_client: httpx.AsyncClient,
                                  id: str,
                                  fields: List[str],
                                  schema: str,
                                  timeout: int) -> GetBatchDocumentResponse:
        async with semaphore:
            try:
                resp = await async_client.get(
                    f'{self.document_url}/document/v1/{schema}/{schema}/docid/{id}?fieldSet={schema}:{",".join(fields)}', timeout=timeout
                )
            except httpx.HTTPError as e:
                raise VespaError(e) from e

            if resp.status_code in [200, 404]:
                return GetBatchDocumentResponse(**resp.json(), status=resp.status_code)

            self._raise_for_status(resp)


    async def _delete_batch_async(self,
                                  ids: List[str],
                                  schema: str,
                                  connections: int, timeout: int) -> DeleteBatchResponse:
        async with httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=connections,
                                                         max_connections=connections)) as async_client:
            semaphore = asyncio.Semaphore(connections)
            tasks = [
                asyncio.create_task(
                    self._delete_document_async(semaphore, async_client, id, schema, timeout)
                )
                for id in ids
            ]
            await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

        responses = []
        errors = False
        for task in tasks:
            result = task.result()
            responses.append(result)
            if result.status != 200:
                errors = True

        return DeleteBatchResponse(responses=responses, errors=errors)

    async def _delete_document_async(self,
                                     semaphore: asyncio.Semaphore,
                                     async_client: httpx.AsyncClient,
                                     id: str,
                                     schema: str,
                                     timeout: int) -> DeleteBatchDocumentResponse:
        async with semaphore:
            try:
                resp = await async_client.delete(f'{self.document_url}/document/v1/{schema}/{schema}/docid/{id}')
            except httpx.HTTPError as e:
                raise VespaError(e) from e

        try:
            # This will cover 200 and document-specific errors. Other unexpected errors will be raised.
            return DeleteBatchDocumentResponse(**resp.json(), status=resp.status_code)
        except JSONDecodeError:
            if resp.status_code == 200:
                # A 200 response shouldn't reach here
                raise VespaError(f'Unexpected response: {resp.text}')

            self._raise_for_status(resp)

    @classmethod
    def _is_timeout_error(cls, error: Error, resp: httpx.Response) -> bool:
        """
        Check if the query error is a timeout error.
        """

        if error.code == 8 and error.message == "Search request soft doomed during query setup and initialization.":
            logger.warn('Detected soft doomed query')
            return True
        if error.code == 12 and resp.status_code == 504:
            return True

        return False

    def _query_raise_for_status(self, resp: httpx.Response) -> None:
        """
        Query API specific raise for status method.
        If multiple errors:
            If all errors are timeout, raise VespaTimeoutError (504).
            If even one error is not timeout, raise VespaStatusError (500).
        """
        # See error codes here https://github.com/vespa-engine/vespa/blob/master/container-core/src/main/java/com/yahoo/container/protect/Error.java
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            try:
                result = QueryResult(**resp.json())
                if (
                        result.root.errors is not None
                        and len(result.root.errors) > 0
                ):
                    for error in result.root.errors:
                        if not self._is_timeout_error(error, resp):
                            # Raise 500 if any error is not timeout
                            raise VespaStatusError(message=resp.text, cause=e) from e
                    # Raise 504 if all errors are timeout
                    raise VespaTimeoutError(message=resp.text, cause=e) from e
                raise e
            except VespaStatusError:
                raise
            except Exception:
                raise VespaStatusError(message=resp.text, cause=e) from e

    def _raise_for_status(self, resp: httpx.Response) -> None:
        """Take the response and raise an VespaStatusError if the status code is not 2xx.

        Args:
            resp: The response object from the httpx client

        Raises:
            VespaStatusError: If the status code is not 2xx
        """
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            response = e.response
            try:
                json = response.json()
                error_code = json['error-code']
                message = json['message']
            except Exception:
                raise VespaStatusError(message=response.text, cause=e) from e

            self._raise_for_error_code(error_code, message, e)

    def _raise_for_error_code(self, error_code: str, message: str, cause: Exception) -> None:
        exception = self._VESPA_ERROR_CODE_TO_EXCEPTION.get(error_code, VespaError)
        if exception:
            raise exception(message=message, cause=cause) from cause

        raise VespaStatusError(message=f'{error_code}: {message}', cause=cause) from cause