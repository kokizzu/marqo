import functools
import math
import os
import uuid
from unittest import mock

import pytest

from marqo.api.exceptions import IndexNotFoundError, BadRequestError
from marqo.core.models.marqo_index import *
from marqo.tensor_search import enums
from marqo.tensor_search import tensor_search
from marqo.core.models.add_docs_params import AddDocsParams
from integ_tests.marqo_test import MarqoTestCase, TestImageUrls


class TestAddDocumentsSemiStructured(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        default_text_index = cls.unstructured_marqo_index_request()
        default_text_index_encoded_name = cls.unstructured_marqo_index_request(
            name='a-b_' + str(uuid.uuid4()).replace('-', '')
        )

        default_image_index = cls.unstructured_marqo_index_request(
            model=Model(name='ViT-B/32'),
            treat_urls_and_pointers_as_images=True
        )

        image_index_with_chunking = cls.unstructured_marqo_index_request(
            model=Model(name='ViT-B/32'),
            image_preprocessing=ImagePreProcessing(patch_method=PatchMethod.Frcnn),
            treat_urls_and_pointers_as_images=True
        )

        image_index_with_random_model = cls.unstructured_marqo_index_request(
            model=Model(name='random'),
            treat_urls_and_pointers_as_images=True
        )

        cls.indexes = cls.create_indexes([
            default_text_index,
            default_text_index_encoded_name,
            default_image_index,
            image_index_with_chunking,
            image_index_with_random_model
        ])

        cls.default_text_index = default_text_index.name
        cls.default_text_index_encoded_name = default_text_index_encoded_name.name
        cls.default_image_index = default_image_index.name
        cls.image_index_with_chunking = image_index_with_chunking.name
        cls.image_index_with_random_model = image_index_with_random_model.name

    def setUp(self) -> None:
        self.clear_indexes(self.indexes)

        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()

    def test_add_plain_id_field(self):
        """
        Plain id field works
        """
        tests = [
            (self.default_text_index, 'Standard index name'),
            (self.default_text_index_encoded_name, 'Index name requiring encoding'),
        ]
        for index_name, desc in tests:
            with self.subTest(desc):
                self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.default_text_index,
                        docs=[{
                            "_id": "123",
                            "title": "content 1",
                            "desc": "content 2. blah blah blah"
                        }],
                        device="cpu", tensor_fields=["title"]
                    )
                )
                self.assertEqual(
                    {
                        "_id": "123",
                        "title": "content 1",
                        "desc": "content 2. blah blah blah"
                    },
                    tensor_search.get_document_by_id(
                        config=self.config, index_name=self.default_text_index,
                        document_id="123"
                    )
                )

    def test_add_documents_dupe_ids(self):
        """
        Only the latest added document is returned
        """

        # Add once to get vectors
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index, docs=[{
                    "_id": "1",
                    "title": "doc 123"
                }],
                device="cpu", tensor_fields=["title"]
            )
        )
        tensor_facets = tensor_search.get_document_by_id(
            config=self.config, index_name=self.default_text_index,
            document_id="1", show_vectors=True)['_tensor_facets']

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index, docs=[
                    {
                        "_id": "2",
                        "title": "doc 000"
                    }
                ],
                device="cpu", tensor_fields=["title"]
            )
        )
        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.default_text_index, docs=[
                    {
                        "_id": "2",
                        "title": "doc 123"
                    }
                ],
                device="cpu", tensor_fields=["title"]
            )
        )

        expected_doc = {
            "_id": "2",
            "title": "doc 123",
            '_tensor_facets': tensor_facets
        }
        actual_doc = tensor_search.get_document_by_id(
            config=self.config, index_name=self.default_text_index,
            document_id="2", show_vectors=True)

        self.assertEqual(expected_doc, actual_doc)

    def test_add_documents_with_missing_index_fails(self):
        rand_index = 'a' + str(uuid.uuid4()).replace('-', '')

        with pytest.raises(IndexNotFoundError):
            self.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=rand_index, docs=[{"abc": "def"}], device="cpu"
                )
            )

    def test_add_documents_whitespace(self):
        """
        Indexing fields consisting of only whitespace works
        """
        docs = [
            {"title": ""},
            {"title": " "},
            {"title": "  "},
            {"title": "\r"},
            {"title": "\r "},
            {"title": "\r\r"},
            {"title": "\r\t\n"},
        ]
        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.default_text_index, docs=docs, device="cpu", tensor_fields=[]
            )
        )
        count = self.pyvespa_client.query(
            {"yql": f"select * from sources {self.default_text_index} where true limit 0"}
        ).json["root"]["fields"]["totalCount"]

        assert count == len(docs)

    def test_add_docs_response_format(self):
        add_res = self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.default_text_index,
                docs=[
                    {
                        "_id": "123",
                        "title": "content 1",
                        "desc": "content 2. blah blah blah"
                    },
                    {
                        "_id": "456",
                        "title": "content 1",
                        "desc": "content 2. blah blah blah"
                    },
                    {
                        "_id": "789",
                        "tags": [1, 'str']  # mixed types, error
                    }
                ],
                device="cpu", tensor_fields=[]
            )
        ).dict(exclude_none=True, by_alias=True)
        assert "errors" in add_res
        assert "processingTimeMs" in add_res
        assert "index_name" in add_res
        assert "items" in add_res

        assert add_res["processingTimeMs"] > 0
        assert add_res["errors"] is True
        assert add_res["index_name"] == self.default_text_index

        for item in add_res["items"]:
            assert "_id" in item
            assert "status" in item
            assert (item['status'] == 200) ^ ("error" in item and "code" in item)

        assert [item['status'] for item in add_res["items"]] == [200, 200, 400]

    def test_add_documents_validation(self):
        """
        Invalid documents return errors
        """
        bad_doc_args = [
            [{"_id": "to_fail_123", "title": ["wow", "this", "is"]}],  # tensor field list
            [{"_id": "to_fail_123", "title": ["wow", "this", "is"]},  # tensor field list
             {"_id": "to_pass_123", "title": 'some_content'}],
            [{"_id": "to_fail_123", "tags": [{"abc": "678"}]}],  # list of dict
            [{"_id": "to_fail_123", "title": {"abc": "234"}}],  # dict for non-combination field
            [{"_id": "to_fail_123", "title": {"abc": "234"}},  # dict for non-combination field
             {"_id": "to_pass_123", "title": 'some_content'}],
            # other checking:
            [{"title": {1243}, "_id": "to_fail_123"}],  # invalid json
            [{"title": None, "_id": "to_fail_123"}],  # None not a valid type
            [{"_id": "to_fail_123", "title": [None], "desc": "123"},  # None not a valid type
             {"_id": "to_fail_567", "title": "finnne", 123: "heehee"}],  # Field name int
            [{"_id": "to_fail_123", "title": [None], "desc": "123"},  # List of None
             {"_id": "to_fail_567", "title": AssertionError}],  # Pointer as value, invalid json
            [{"_id": "to_fail_567", "tags": max}]  # Invalid json
        ]

        # For replace, check with use_existing_tensors True and False
        for use_existing_tensors_flag in (True, False):
            for bad_doc_arg in bad_doc_args:
                with self.subTest(msg=f'{bad_doc_arg} - use_existing_tensors={use_existing_tensors_flag}'):
                    add_res = self.add_documents(
                        config=self.config, add_docs_params=AddDocsParams(
                            index_name=self.default_text_index, docs=bad_doc_arg,
                            use_existing_tensors=use_existing_tensors_flag, device="cpu",
                            tensor_fields=["title"]
                        )
                    ).dict(exclude_none=True, by_alias=True)
                    assert add_res['errors'] is True
                    assert all(['error' in item for item in add_res['items'] if item['_id'].startswith('to_fail')])
                    assert all([item['status'] == 200
                                for item in add_res['items'] if item['_id'].startswith('to_pass')])

    def test_add_documents_id_validation(self):
        """
        Invalid document IDs return errors
        """
        bad_doc_args = [
            # Wrong data types for ID
            # Tuple: (doc_list, number of docs that should succeed)
            ([{"_id": {}, "title": "yyy"}], 0),
            ([{"_id": dict(), "title": "yyy"}], 0),
            ([{"_id": [1, 2, 3], "title": "yyy"}], 0),
            ([{"_id": 4, "title": "yyy"}], 0),
            ([{"_id": None, "title": "yyy"}], 0),
            ([{"_id": "proper id", "title": "yyy"},
              {"_id": ["bad", "id"], "title": "zzz"},
              {"_id": "proper id 2", "title": "xxx"}], 2)
        ]

        # For replace, check with use_existing_tensors True and False
        for use_existing_tensors_flag in (True, False):
            for bad_doc_arg in bad_doc_args:
                with self.subTest(f'{bad_doc_arg} - use_existing_tensors={use_existing_tensors_flag}'):
                    add_res = self.add_documents(
                        config=self.config, add_docs_params=AddDocsParams(
                            index_name=self.default_text_index, docs=bad_doc_arg[0],
                            use_existing_tensors=use_existing_tensors_flag, device="cpu", tensor_fields=["title"]
                        )
                    ).dict(exclude_none=True, by_alias=True)
                    assert add_res[
                               'errors'] is True, f'{bad_doc_arg} - use_existing_tensors={use_existing_tensors_flag}'
                    succeeded_count = 0
                    for item in add_res['items']:
                        if item['status'] == 200:
                            succeeded_count += 1
                        else:
                            assert 'Document _id must be a string type' in item['error']

                    assert succeeded_count == bad_doc_arg[1]

    def test_add_documents_list_success(self):
        good_docs = [
            [{"_id": "to_fail_123", "tags": ["wow", "this", "is"]}]
        ]
        for bad_doc_arg in good_docs:
            add_res = self.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.default_text_index,
                    docs=bad_doc_arg,
                    device="cpu",
                    tensor_fields=[],
                )
            ).dict(exclude_none=True, by_alias=True)
            assert add_res['errors'] is False

    def test_add_documents_list_data_type_validation(self):
        """These bad docs should return errors"""
        self.tags_ = [
            [{"_id": "to_fail_123", "tags": ["wow", "this", False]}],
            [{"_id": "to_fail_124", "tags": [1, None, 3]}],
            [{"_id": "to_fail_125", "tags": [{}]}],
            [{"_id": "to_fail_126", "tags": [1, 2, 3]}],
            [{"_id": "to_fail_127", "tags": [1.0, 2.0, 3.0]}],
            [{"_id": "to_fail_128", "tags": [1, 2.0, 3]}],
        ]
        bad_doc_args = self.tags_
        for bad_doc_arg in bad_doc_args:
            with self.subTest(bad_doc_arg):
                add_res = self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.default_text_index,
                        docs=bad_doc_arg,
                        device="cpu",
                        tensor_fields=[],
                    )
                ).dict(exclude_none=True, by_alias=True)
                assert add_res['errors'] is True
                assert all(['error' in item for item in add_res['items']])
                assert all(['Unstructured Marqo index only supports string lists.' in item['message']
                            for item in add_res['items']])

    def test_add_documents_set_device(self):
        """
        Device is set correctly
        """
        mock_vectorise = mock.MagicMock()
        mock_vectorise.return_value = [[0, 0, 0, 0]]

        @mock.patch("marqo.s2_inference.s2_inference.vectorise", mock_vectorise)
        def run():
            self.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.default_text_index, device="cuda:22", docs=[{"title": "doc"}, {"title": "doc"}],
                    tensor_fields=["title"]
                ),
            )
            return True

        assert run()
        args, kwargs = mock_vectorise.call_args
        assert kwargs["device"] == "cuda:22"

    def test_add_documents_empty(self):
        """
        Adding empty documents raises BadRequestError
        """
        try:
            self.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.default_text_index, docs=[],
                    device="cpu")
            )
            raise AssertionError
        except BadRequestError:
            pass

    def test_add_documents_id_image_url(self):
        """
        Image URL as ID is not downloaded
        """
        docs = [{
            "_id": TestImageUrls.HIPPO_REALISTIC.value,
            "title": "wow"}
        ]

        with mock.patch('PIL.Image.open') as mock_image_open:
            self.add_documents(config=self.config,
                               add_docs_params=AddDocsParams(
                                            index_name=self.default_image_index, docs=docs,
                                            device="cpu", tensor_fields=["title"]
                                        ))

            mock_image_open.assert_not_called()

    def test_add_documents_resilient_doc_validation(self):
        docs_results = [
            # handle empty dicts
            ([{"_id": "123", "title": "legitimate text"},
              {},
              {"_id": "456", "title": "awesome stuff!"}],
             [("123", 200), (None, 400), ('456', 200)]
             ),
            ([{}], [(None, 400)]),
            ([{}, {}], [(None, 400), (None, 400)]),
            ([{}, {}, {"title": "yep"}], [(None, 400), (None, 400), (None, 200)]),
            # handle invalid dicts
            ([{"this is a set, lmao"}, "this is a string", {"title": "yep"}],
             [(None, 400), (None, 400), (None, 200)]),
            ([1234], [(None, 400)]),
            ([None], [(None, 400)]),
            # handle invalid field names
            ([{123: "bad"}, {"_id": "cool"}], [(None, 400), ("cool", 200)]),
            ([{"__chunks": "bad"}, {"_id": "1511", "__vector_a": "some content"}, {"_id": "cool"},
              {"_id": "144451", "__field_content": "some content"}],
             [(None, 200), ("1511", 200), ("cool", 200), ("144451", 200)]),
            ([{123: "bad", "_id": "12345"}, {"_id": "cool"}], [("12345", 400), ("cool", 200)]),
            ([{None: "bad", "_id": "12345"}, {"_id": "cool"}], [("12345", 400), ("cool", 200)]),
            # handle bad content
            ([{"title": None, "_id": "12345"}, {"_id": "cool"}], [(None, 400), ("cool", 200)]),
            ([{"tags": [1, 2, '3', 4], "_id": "12345"}, {"_id": "cool"}], [("12345", 400), ("cool", 200)]),
            ([{"title": ("cat", "dog"), "_id": "12345"}, {"_id": "cool"}], [("12345", 400), ("cool", 200)]),
            ([{"title": set(), "_id": "12345"}, {"_id": "cool"}], [(None, 400), ("cool", 200)]),
            # handle bad _ids
            ([{"bad": "hehehe", "_id": 12345}, {"_id": "cool"}], [(None, 400), ("cool", 200)]),
            ([{"bad": "hehehe", "_id": 12345}, {"_id": "cool"}, {"bad": "hehehe", "_id": None}, {"title": "yep"},
              {"_id": (1, 2), "efgh": "abc"}, {"_id": 1.234, "cool": "wowowow"}],
             [(None, 400), ("cool", 200), (None, 400), (None, 200), (None, 400),
              (None, 400)]),
            # mixed
            ([{(1, 2, 3): set(), "_id": "12345"}, {"_id": "cool"}, {"tags": [1, 2, 3], "_id": None}, {"title": "yep"},
              {}, "abcdefgh"],
             [(None, 400), ("cool", 200), (None, 400), (None, 200), (None, 400),
              (None, 400)]),
        ]
        for docs, expected_results in docs_results:
            with self.subTest(f'{expected_results}'):
                add_res = self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.default_text_index, docs=docs,
                        device="cpu", tensor_fields=[]
                    )
                ).dict(exclude_none=True, by_alias=True)
                self.assertEqual(len(expected_results), len(expected_results))
                for i, res_dict in enumerate(add_res['items']):
                    # if the expected id is None, then it assumed the id is
                    # generated and can't be asserted against
                    if expected_results[i][0] is not None:
                        self.assertEqual(expected_results[i][0], res_dict["_id"])
                    self.assertEqual(expected_results[i][1], res_dict['status'])

    def test_add_document_with_tensor_fields(self):
        """Ensure tensor_fields only works for title but not desc"""
        docs_ = [{"_id": "789", "title": "Story of Alice Appleseed", "desc": "Alice grew up in Houston, Texas."}]
        self.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.default_text_index, docs=docs_, device="cpu", tensor_fields=["title"]
        ))
        resp = tensor_search.get_document_by_id(config=self.config,
                                                index_name=self.default_text_index, document_id="789",
                                                show_vectors=True)

        assert len(resp[enums.TensorField.tensor_facets]) == 1
        assert enums.TensorField.embedding in resp[enums.TensorField.tensor_facets][0]
        assert "title" in resp[enums.TensorField.tensor_facets][0]
        assert "desc" not in resp[enums.TensorField.tensor_facets][0]

    def test_doc_too_large(self):
        max_size = 400000
        mock_environ = {enums.EnvVars.MARQO_MAX_DOC_BYTES: str(max_size)}

        @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
        def run():
            update_res = self.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.default_text_index, docs=[
                        {"_id": "123", 'desc': "edf " * (max_size // 4)},
                        {"_id": "789", "desc": "abc " * ((max_size // 4) - 500)},
                        {"_id": "456", "desc": "exc " * (max_size // 4)},
                    ],
                    device="cpu", tensor_fields=["desc"]
                )).dict(exclude_none=True, by_alias=True)
            items = update_res['items']
            assert update_res['errors']
            assert 'error' in items[0] and 'error' in items[2]
            assert 'doc_too_large' == items[0]['code'] and ('doc_too_large' == items[0]['code'])
            assert items[1]['status'] == 200
            assert 'error' not in items[1]
            return True

        assert run()

    def test_doc_too_large_single_doc(self):
        max_size = 400000
        mock_environ = {enums.EnvVars.MARQO_MAX_DOC_BYTES: str(max_size)}

        @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
        def run():
            update_res = self.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.default_text_index, docs=[
                        {"_id": "123", 'desc': "edf " * (max_size // 4)},
                    ],
                    use_existing_tensors=True, device="cpu", tensor_fields=[])
            ).dict(exclude_none=True, by_alias=True)
            items = update_res['items']
            assert update_res['errors']
            assert 'error' in items[0]
            assert 'doc_too_large' == items[0]['code']
            return True

        assert run()

    def test_doc_too_large_none_env_var(self):
        """
        If MARQO_MAX_DOC_BYTES is not set, then the default is used
        """
        # TODO - Consider removing this test as indexing a standard doc is covered by many other tests
        for env_dict in [dict()]:
            @mock.patch.dict(os.environ, {**os.environ, **env_dict})
            def run():
                update_res = self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.default_text_index, docs=[
                            {"_id": "123", 'desc': "Some content"},
                        ],
                        use_existing_tensors=True, device="cpu", tensor_fields=["desc"]
                    )).dict(exclude_none=True, by_alias=True)
                items = update_res['items']
                assert not update_res['errors']
                assert 'error' not in items[0]
                assert items[0]['status'] == 200
                return True

            assert run()

    def test_remove_tensor_field(self):
        """
        If a document is re-indexed with a tensor field removed, the vectors are removed
        """
        # test replace and update workflows
        self.add_documents(
            self.config, add_docs_params=AddDocsParams(
                docs=[{"_id": "123", "title": "mydata", "desc": "mydata2"}],
                index_name=self.default_text_index, device="cpu", tensor_fields=["title"]
            )
        )
        self.add_documents(
            self.config,
            add_docs_params=AddDocsParams(
                docs=[{"_id": "123", "desc": "mydata"}],
                index_name=self.default_text_index,
                device="cpu",
                tensor_fields=[]
            )
        )
        doc_w_facets = tensor_search.get_document_by_id(
            self.config, index_name=self.default_text_index, document_id='123', show_vectors=True)
        assert doc_w_facets[enums.TensorField.tensor_facets] == []
        assert 'title' not in doc_w_facets

    def test_add_documents_exceeded_max_doc_count(self):
        max_docs = 128

        test_cases = [  # count, error out=?
            (max_docs - 10, False),
            (max_docs - 1, False),
            (max_docs, False),
            (max_docs + 1, True),
            (max_docs + 10, True),
        ]

        for count, error in test_cases:
            with self.subTest(f'{count} - {error}'):

                if error:
                    with self.assertRaises(BadRequestError):
                        self.add_documents(
                            config=self.config, add_docs_params=AddDocsParams(
                                index_name=self.default_text_index,
                                docs=[{
                                    "desc": "some desc"
                                }] * count,
                                tensor_fields=[],
                                device="cpu"
                            )
                        )
                else:
                    self.assertEqual(False,
                                     self.add_documents(
                                         config=self.config, add_docs_params=AddDocsParams(
                                             index_name=self.default_text_index,
                                             docs=[{
                                                 "desc": "some desc"
                                             }] * count,
                                             tensor_fields=[],
                                             device="cpu"
                                         )
                                     ).dict(exclude_none=True, by_alias=True)['errors']
                                     )

    def test_no_tensor_field_on_empty_ix(self):
        """
        If a document is indexed with no tensor fields on an empty index, no vectors are added
        """
        self.add_documents(
            self.config, add_docs_params=AddDocsParams(
                docs=[{"_id": "123", "desc": "mydata"}],
                index_name=self.default_text_index,
                device="cpu", tensor_fields=[]
            )
        )
        doc_w_facets = tensor_search.get_document_by_id(
            self.config, index_name=self.default_text_index, document_id='123', show_vectors=True)
        assert doc_w_facets[enums.TensorField.tensor_facets] == []
        assert 'desc' in doc_w_facets

    def test_index_doc_on_empty_ix(self):
        """
        If a document is indexed with a tensor field vectors are added for the tensor field
        """
        self.add_documents(
            self.config, add_docs_params=AddDocsParams(
                docs=[{"_id": "123", "title": "mydata", "desc": "mydata"}],
                index_name=self.default_text_index, tensor_fields=["title"],
                device="cpu"
            )
        )
        doc_w_facets = tensor_search.get_document_by_id(
            self.config, index_name=self.default_text_index, document_id='123', show_vectors=True)
        assert len(doc_w_facets[enums.TensorField.tensor_facets]) == 1
        assert 'title' in doc_w_facets[enums.TensorField.tensor_facets][0]
        assert 'desc' not in doc_w_facets[enums.TensorField.tensor_facets][0]
        assert 'title' in doc_w_facets
        assert 'desc' in doc_w_facets

    def test_various_image_count(self):
        hippo_url = TestImageUrls.HIPPO_REALISTIC.value

        def _check_get_docs(doc_count, title_value):
            approx_half = math.floor(doc_count / 2)
            get_res = tensor_search.get_documents_by_ids(
                config=self.config, index_name=self.image_index_with_random_model,
                document_ids=[str(n) for n in (0, approx_half, doc_count - 1)],
                show_vectors=True
            ).dict(exclude_none=True, by_alias=True)
            for d in get_res['results']:
                assert d['_found'] is True
                assert d['title'] == title_value
                assert d['location'] == hippo_url
                assert {'_embedding', 'location', 'title'} == functools.reduce(lambda x, y: x.union(y),
                                                                               [list(facet.keys()) for facet in
                                                                                d['_tensor_facets']], set())
                for facet in d['_tensor_facets']:
                    if 'location' in facet:
                        assert facet['location'] == hippo_url
                    elif 'title' in facet:
                        assert facet['title'] == title_value
                    assert isinstance(facet['_embedding'], list)
                    assert len(facet['_embedding']) > 0
            return True

        doc_counts = 1, 2, 25
        for c in doc_counts:
            self.clear_index_by_index_name(self.image_index_with_random_model)

            res1 = self.add_documents(
                self.config,
                add_docs_params=AddDocsParams(
                    docs=[{"_id": str(doc_num),
                           "location": hippo_url,
                           "title": "blah"} for doc_num in range(c)],
                    index_name=self.image_index_with_random_model, device="cpu",
                    tensor_fields=["title", "location"]
                )
            ).dict(exclude_none=True, by_alias=True)
            print(res1)
            self.assertEqual(
                c,
                self.config.monitoring.get_index_stats_by_name(
                    index_name=self.image_index_with_random_model
                ).number_of_documents,
            )
            self.assertFalse(res1['errors'])
            self.assertTrue(_check_get_docs(doc_count=c, title_value='blah'))

    def test_bad_tensor_fields(self):
        test_cases = [
            ({"tensor_fields": None}, "tensor_fields must be explicitly provided", "None as tensor fields"),
            ({}, "tensor_fields must be explicitly provided", "No tensor fields"),
            ({"tensor_fields": ["_id", "some"]}, "`_id` field cannot be a tensor field", "_id can't be a tensor field")
        ]
        for tensor_fields, error_message, msg in test_cases:
            with self.subTest(msg):
                with self.assertRaises(BadRequestError) as e:
                    self.add_documents(
                        config=self.config,
                        add_docs_params=AddDocsParams(index_name=self.default_text_index,
                                                      docs=[{"some": "data"}], **tensor_fields))
                self.assertIn(error_message, e.exception.message)

    def test_supported_large_integer_and_float_number(self):
        """Test to ensure large integer and float numbers are handled correctly for long and double data types
        in the unstructured index schema"""
        test_case = [
            ({"long_field_1": 1}, False),  # small positive integer
            ({"long_field_1": -1}, False),  # small negative integer
            ({"long_field_1": -100232142002}, False),  # large positive integer that can't be handled by int
            ({"long_field_1": -923217213231}, False),  # large negative integer that can't be handled by int
            ({'long_field_1': int("1" * 50)}, True),  # overlarge positive integer, should raise error in long field
            # overlarge negative integer, should raise error in long field
            ({'long_field_1': -1 * int("1" * 50)}, True),
            ({"double_field_1": 1e10}, False),  # large positive integer mathematical expression
            ({"double_field_1": -1e12}, False),  # large negative integer mathematical expression
            ({"double_field_1": 1e10 + 0.123249357987123}, False),  # large positive float
            ({"double_field_1": - 1e10 + 0.123249357987123}, False),  # large negative float
        ]

        for doc, error in test_case:
            with self.subTest():
                res = self.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.default_text_index, docs=[doc], device="cpu",
                        tensor_fields=[]
                    )
                ).dict(exclude_none=True, by_alias=True)
                print(res)
                self.assertEqual(res['errors'], error)

    @pytest.mark.skip_for_multinode
    def test_duplicate_ids_behaviour(self):
        """Test the behaviour when there are duplicate ids in a single batch.

        Note: The expected behaviour is that the last document given in the batch is used while the formers are ignored.
        """

        test_cases = [
            ([{"_id": "1", "text_field": "test 1"}, {"_id": "1", "text_field": "test 2"}], 1, "Normal case"),
            ([{"_id": "1", "text_field": "test 1"}, {"_id": "1", "text_field::": "test 2"}], 0,
             "Even if the last document is invalid, it should be used"),
            ([{"_id": "1", "text_field::": "test 2"}, {"_id": "1", "text_field": "test 1"}], 1,
             "If the previous document is invalid, it should not affect the last document"),
        ]

        for documents, number_of_docs, msg in test_cases:
            self.clear_index_by_index_name(self.default_text_index)
            with self.subTest(msg):
                r = self.add_documents(config=self.config,
                                       add_docs_params=AddDocsParams(
                                                    index_name=self.default_text_index, docs=documents,
                                                    device="cpu", tensor_fields=["text_field"]
                                                )).dict(exclude_none=True, by_alias=True)
                self.assertEqual(1, len(r["items"]))
                number_of_docs_in_index = self.config.monitoring.get_index_stats_by_name(
                    index_name=self.default_text_index).number_of_documents
                self.assertEqual(number_of_docs, number_of_docs_in_index)

    def test_a_text_index_will_treat_a_url_as_text(self):
        """Test that a text index will treat a URL as text and not download the image"""
        valid_url = TestImageUrls.HIPPO_REALISTIC.value
        invalid_url = TestImageUrls.HIPPO_REALISTIC.value + "invalid"
        self.add_documents(
            config=self.config, add_docs_params=AddDocsParams(
                index_name=self.default_text_index, docs=[
                    {
                        "_id": "1",
                        "title": invalid_url,
                        "non_tensor_field": valid_url
                    }
                ],
                device="cpu", tensor_fields=["title"]
            )
        )
        doc = tensor_search.get_document_by_id(
            config=self.config, index_name=self.default_text_index, document_id="1", show_vectors=True
        )

        self.assertEqual(invalid_url, doc["title"])
        self.assertEqual(valid_url, doc["non_tensor_field"])
        self.assertEqual(1, len(doc[enums.TensorField.tensor_facets]))
        self.assertIn("title", doc[enums.TensorField.tensor_facets][0])