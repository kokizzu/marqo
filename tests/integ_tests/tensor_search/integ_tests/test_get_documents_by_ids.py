import functools
import os
import uuid
from unittest import mock
from unittest.mock import patch

from integ_tests.marqo_test import MarqoTestCase
from integ_tests.utils.transition import *

from marqo.api.exceptions import (
    InvalidArgError,
    IllegalRequestedDocCount
)
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import FieldRequest
from marqo.tensor_search import enums
from marqo.tensor_search import tensor_search


class TestGetDocuments(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        structured_text_index_with_random_model_request = cls.structured_marqo_index_request(
            model=Model(name='random'),
            fields=[
                FieldRequest(name='title1', type=FieldType.Text),
                FieldRequest(name='desc2', type=FieldType.Text),
                FieldRequest(name='int_field', type=FieldType.Int),
                FieldRequest(name='int_array_field', type=FieldType.ArrayInt),
                FieldRequest(name='int_map_field', type=FieldType.MapInt),
                FieldRequest(name='float_field', type=FieldType.Float),
                FieldRequest(name='float_array_field', type=FieldType.ArrayFloat),
                FieldRequest(name='float_map_field', type=FieldType.MapFloat),
                FieldRequest(name='long_field', type=FieldType.Long),
                FieldRequest(name='long_array_field', type=FieldType.ArrayLong),
                FieldRequest(name='long_map_field', type=FieldType.MapLong),
                FieldRequest(name='double_field', type=FieldType.Double),
                FieldRequest(name='double_array_field', type=FieldType.ArrayDouble),
                FieldRequest(name='double_map_field', type=FieldType.MapDouble),
                FieldRequest(name='string_array_field', type=FieldType.ArrayText),
                FieldRequest(name='bool_field', type=FieldType.Bool),
                FieldRequest(name='custom_vector_field', type=FieldType.CustomVector),
            ],
            tensor_fields=["title1", "desc2", "custom_vector_field"]
        )
        unstructured_text_index_with_random_model_request = cls.unstructured_marqo_index_request(
            model=Model(name='random'),
            marqo_version='2.12.0'
        )
        semi_structured_text_index_with_random_model_request = cls.unstructured_marqo_index_request(
            model=Model(name='random'),
        )

        # List of indexes to loop through per test. Test itself should extract index name.
        cls.indexes = cls.create_indexes([
            structured_text_index_with_random_model_request,
            unstructured_text_index_with_random_model_request,
            semi_structured_text_index_with_random_model_request
        ])

    def setUp(self) -> None:
        # Any tests that call add_documents, search, bulk_search need this env var
        self.device_patcher = patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

    def tearDown(self) -> None:
        self.device_patcher.stop()

    def test_get_documents_by_ids_structured(self):
        index = self.indexes[0]

        docs = [
            {"_id": "1", "title1": "content 1",
             "int_field": 1, "int_array_field": [1, 2], "int_map_field": {"a": 1},
             "float_field": 2.9, "float_array_field": [1.0, 2.0], "float_map_field": {"b": 2.9},
             "long_field": 10, "long_array_field": [10, 20], "long_map_field": {"a": 10},
             "double_field": 3.9, "double_array_field": [3.0, 5.0], "double_map_field": {"b": 5.9},
             "bool_field": True, "string_array_field": ["a", "b", "c"]},
            {"_id": "2", "title1": "content 2", "custom_vector_field": {"content": "a", "vector": [1.0] * 384}},
            {"_id": "3", "title1": "content 3"}
        ]

        self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=index.name, docs=docs, device="cpu"
            )
        )
        res = tensor_search.get_documents_by_ids(
            config=self.config, index_name=index.name, document_ids=['1', '2', '3'],
            show_vectors=True).dict(exclude_none=True, by_alias=True)

        # Check that the documents are found and have the correct content
        for i in range(3):
            self.assertEqual(res['results'][i]['_found'], True)

            for field_name, value in res['results'][i].items():
                if field_name in [enums.TensorField.tensor_facets, "_found"]:
                    # ignore meta fields
                    continue
                if field_name == "custom_vector_field":
                    expected_value = docs[i]["custom_vector_field"]["content"]
                else:
                    expected_value = docs[i][field_name]

                self.assertEqual(expected_value, value)

            self.assertIn(enums.TensorField.tensor_facets, res['results'][i])
            self.assertIn(enums.TensorField.embedding, res['results'][i][enums.TensorField.tensor_facets][0])

    def test_get_documents_by_ids_unstructured(self):
        for index in self.indexes:
            if index.type == IndexType.Structured:
                continue

            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                docs = [
                    {"_id": "1", "title1": "content 1", "int_field": 1, "int_map_field": {"a": 1}, "float_field": 2.9,
                     "float_map_field": {"b": 2.9}, "bool_field": True, "string_array_field": ["a", "b", "c"]},
                    {"_id": "2", "title1": "content 2", "custom_vector_field": {"content": "a", "vector": [1.0] * 384}},
                    {"_id": "3", "title1": "content 3"}
                ]
                self.add_documents(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=index.name, docs=docs, device="cpu",
                        tensor_fields=["title1", "desc2", "custom_vector_field"],
                        mappings={"custom_vector_field": {"type": "custom_vector"}}
                    )
                )
                res = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=index.name, document_ids=['1', '2', '3'],
                    show_vectors=True).dict(exclude_none=True, by_alias=True)

                # Check that the documents are found and have the correct content
                for i in range(3):
                    self.assertEqual(res['results'][i]['_found'], True)

                    for field_name, value in res['results'][i].items():
                        if field_name in [enums.TensorField.tensor_facets, "_found"]:
                            # ignore meta fields
                            continue
                        if field_name == "custom_vector_field":
                            expected_value = docs[i]["custom_vector_field"]["content"]
                        elif '.' in field_name:
                            # unstructured and semi-structured indexes have all map fields flattened
                            map_field_name, key = field_name.split('.', 1)
                            expected_value = docs[i][map_field_name][key]
                        else:
                            expected_value = docs[i][field_name]

                        self.assertEqual(expected_value, value)

                    self.assertIn(enums.TensorField.tensor_facets, res['results'][i])
                    self.assertIn(enums.TensorField.embedding, res['results'][i][enums.TensorField.tensor_facets][0])

    def test_get_documents_vectors_format(self):
        keys = [("title1", "desc2", "_id"), ("title1", "desc2", "_id")]
        vals = [("content 1", "content 2. blah blah blah", "123"),
                ("some more content", "some cool desk", "5678")]

        for index in self.indexes:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                self.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=index.name, docs=[dict(zip(k, v)) for k, v in zip(keys, vals)],
                    device="cpu",
                    tensor_fields=["title1", "desc2"] if isinstance(index, UnstructuredMarqoIndex) else None))
                get_res = tensor_search.get_documents_by_ids(
                    config=self.config, index_name=index.name,
                    document_ids=["123", "5678"], show_vectors=True).dict(exclude_none=True, by_alias=True)['results']
                self.assertEqual(2, len(get_res))
                for i, retrieved_doc in enumerate(get_res):
                    assert enums.TensorField.tensor_facets in retrieved_doc
                    assert len(retrieved_doc[enums.TensorField.tensor_facets]) == 2
                    assert set(keys[i]).union({enums.TensorField.embedding}) - {'_id'} == functools.reduce(
                        lambda x, y: x.union(y),
                        [set(facet.keys()) for facet in retrieved_doc[enums.TensorField.tensor_facets]]
                    )
                    for facet in retrieved_doc[enums.TensorField.tensor_facets]:
                        assert len(facet) == 2
                        if keys[0] in facet:
                            assert facet[keys[0]] == vals[0]
                        if keys[1] in facet:
                            assert facet[keys[1]] == vals[1]
                        assert enums.TensorField.embedding in facet

    def test_get_document_vectors_non_existent(self):
        id_reqs = [
            ['123', '456'], ['124']
        ]

        for index in self.indexes:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                for is_vector_shown in (True, False):
                    for i, ids in enumerate(id_reqs):
                        res = tensor_search.get_documents_by_ids(
                            config=self.config, index_name=index.name, document_ids=ids,
                            show_vectors=is_vector_shown
                        ).dict(exclude_none=True, by_alias=True)
                        assert {ii['_id'] for ii in res['results']} == set(id_reqs[i])
                        for doc_res in res['results']:
                            assert not doc_res['_found']

    def test_get_document_vectors_resilient(self):
        for index in self.indexes:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                self.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=index.name, docs=[
                        {"_id": '456', "title1": "alexandra"},
                        {'_id': '221', 'desc2': 'hello'}],
                    device="cpu",
                    tensor_fields=["title1", "desc2"] if isinstance(index, UnstructuredMarqoIndex) else None)
                                   )
                id_reqs = [
                    (['123', '456'], [False, True]), ([['456', '789'], [True, False]]),
                    ([['456', '789', '221'], [True, False, True]]), ([['vkj', '456', '4891'], [False, True, False]])
                ]
                for is_vector_shown in (True, False):
                    for i, (ids, presence) in enumerate(id_reqs):
                        res = tensor_search.get_documents_by_ids(
                            config=self.config, index_name=index.name, document_ids=ids,
                            show_vectors=is_vector_shown
                        ).dict(exclude_none=True, by_alias=True)
                        assert [ii['_id'] for ii in res['results']] == id_reqs[i][0]
                        for j, doc_res in enumerate(res['results']):
                            assert doc_res['_id'] == ids[j]
                            assert doc_res['_found'] == presence[j]
                            if doc_res['_found'] and is_vector_shown:
                                assert enums.TensorField.tensor_facets in doc_res
                                assert 'title1' in doc_res or 'desc2' in doc_res

    def test_get_documents_by_ids_RaiseErrorWithWrongIds(self):
        test_cases = [
            (None, "None is not a valid document id"),
            (dict(), "dict() is not a valid document id"),
            (123, "integer is not a valid document id"),
            (1.23, "float is not a valid document id"),
            ([], "empty list is not a valid document id"),
        ]
        for index in self.indexes:
            for show_vectors_option in (True, False):
                for document_ids, msg in test_cases:
                    with self.subTest(f"Index type: {index.type}. Index name: {index.name}. Msg: {msg}"):
                        with self.assertRaises(InvalidArgError) as e:
                            tensor_search.get_documents_by_ids(
                                config=self.config, index_name=index.name, document_ids=document_ids,
                                show_vectors=show_vectors_option
                            )
                            if not document_ids == []:
                                self.assertIn("Get documents must be passed a collection of IDs!",
                                              str(e.exception))
                            else:
                                self.assertIn("Can't get empty collection of IDs!", str(e.exception))

    def test_get_documents_by_ids_InvalidIdsResponse(self):
        test_cases = [
            (["123", 2], (1,), "2 is not a valid document id"),
            (["123", None], (1,), "None is not a valid document id"),
            ([dict(), 2.3], (0, 1), "dict() and floats not a valid document id"),
        ]
        for index in self.indexes:
            for show_vectors_option in (True, False):
                for document_ids, error_index, msg in test_cases:
                    with self.subTest(f"Index type: {index.type}. Index name: {index.name}. Msg: {msg}"):
                        r = tensor_search.get_documents_by_ids(
                            config=self.config, index_name=index.name, document_ids=document_ids,
                            show_vectors=show_vectors_option
                        )
                        for i in error_index:
                            item = r.results[i]
                            self.assertEqual(item.id, document_ids[i])
                            self.assertEqual(item.status, 400)
                            self.assertIn("Document _id must be a string type!", item.message)
                            self.assertEqual(item.found, None)

    def test_get_documents_env_limit(self):
        for index in self.indexes:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                docs = [{"title1": "a", "_id": uuid.uuid4().__str__()} for _ in range(2000)]
                add_docs_batched(
                    config=self.config,
                    index_name=index.name,
                    docs=docs, device="cpu",
                    tensor_fields=["title1", "desc2"] if isinstance(index, UnstructuredMarqoIndex) else None
                )
                for max_doc in [0, 1, 2, 5, 10, 100, 1000]:
                    mock_environ = {enums.EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: str(max_doc)}

                    @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
                    def run():
                        half_search = tensor_search.get_documents_by_ids(
                            config=self.config, index_name=index.name,
                            document_ids=[docs[i]['_id'] for i in range(max_doc // 2)]
                        ).dict(exclude_none=True, by_alias=True)
                        self.assertEqual(len(half_search['results']), max_doc // 2)
                        limit_search = tensor_search.get_documents_by_ids(
                            config=self.config, index_name=index.name,
                            document_ids=[docs[i]['_id'] for i in range(max_doc)]
                        ).dict(exclude_none=True, by_alias=True)
                        self.assertEqual(len(limit_search['results']), max_doc)
                        with self.assertRaises(IllegalRequestedDocCount):
                            oversized_search = tensor_search.get_documents_by_ids(
                                config=self.config, index_name=index.name,
                                document_ids=[docs[i]['_id'] for i in range(max_doc + 1)]
                            ).dict(exclude_none=True, by_alias=True)
                        with self.assertRaises(IllegalRequestedDocCount):
                            very_oversized_search = tensor_search.get_documents_by_ids(
                                config=self.config, index_name=index.name,
                                document_ids=[docs[i]['_id'] for i in range(max_doc * 2)]
                            ).dict(exclude_none=True, by_alias=True)
                        return True
                assert run()

    def test_limit_results_none(self):
        """if env var isn't set or is None"""
        for index in self.indexes:
            with self.subTest(f"Index type: {index.type}. Index name: {index.name}"):
                docs = [{"title1": "a", "_id": uuid.uuid4().__str__()} for _ in range(2000)]

                add_docs_batched(
                    config=self.config,
                    index_name=index.name,
                    docs=docs, device="cpu",
                    tensor_fields=["title1", "desc2"] if isinstance(index, UnstructuredMarqoIndex) else None
                )

                for mock_environ in [dict(),
                                     {enums.EnvVars.MARQO_MAX_RETRIEVABLE_DOCS: ''}]:
                    @mock.patch.dict(os.environ, {**os.environ, **mock_environ})
                    def run():
                        sample_size = 500
                        limit_search = tensor_search.get_documents_by_ids(
                            config=self.config, index_name=index.name,
                            document_ids=[docs[i]['_id'] for i in range(sample_size)]
                        ).dict(exclude_none=True, by_alias=True)
                        assert len(limit_search['results']) == sample_size
                        return True

                    assert run()
