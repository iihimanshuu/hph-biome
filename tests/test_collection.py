from dataclasses import FrozenInstanceError

import pytest

from hphbiome import (
    CuratedKnowledgeRecord,
    KnowledgeCollection,
    ScientificReference,
)


def make_record(
    record_id: str,
    *,
    references: list[str] | None = None,
) -> CuratedKnowledgeRecord:
    return CuratedKnowledgeRecord(
        id=record_id,
        title=f'Fictional title for {record_id}',
        synthesis=f'Fictional synthesis for {record_id}.',
        references=[] if references is None else references,
        review_status='fictional-review-state',
    )


def make_reference(
    name: str,
    *,
    identifier: str | None = None,
    canonical_url: str | None = None,
) -> ScientificReference:
    return ScientificReference(
        title=f'Fictional source {name}',
        authors=[f'Example Author {name}'],
        source='Imaginary Research Review',
        identifier=identifier,
        canonical_url=canonical_url,
    )


def test_empty_collection_is_accepted() -> None:
    collection = KnowledgeCollection()

    assert collection.records == ()
    assert collection.references == ()


def test_collection_preserves_order_and_detaches_input_sequences() -> None:
    first_record = make_record('fictional-record-b')
    second_record = make_record('fictional-record-a')
    first_reference = make_reference('B', identifier='fictional-id:b')
    second_reference = make_reference(
        'A', canonical_url='https://example.test/sources/a'
    )
    records = [first_record, second_record]
    references = [first_reference, second_reference]

    collection = KnowledgeCollection(
        records=records,
        references=references,
    )
    records.reverse()
    references.reverse()

    assert collection.records == (first_record, second_record)
    assert collection.references == (first_reference, second_reference)
    with pytest.raises(FrozenInstanceError):
        setattr(collection, 'records', ())


def test_collection_looks_up_records_and_reference_aliases() -> None:
    record = make_record('fictional-record-1')
    identifier_reference = make_reference(
        'identifier', identifier='fictional-id:alpha-001'
    )
    url_reference = make_reference(
        'URL', canonical_url='https://example.test/sources/beta-002'
    )
    collection = KnowledgeCollection(
        records=[record],
        references=[identifier_reference, url_reference],
    )

    assert collection.get_record(record.id) is record
    assert (
        collection.get_reference_by_identifier('fictional-id:alpha-001')
        is identifier_reference
    )
    assert (
        collection.get_reference_by_canonical_url(
            'https://example.test/sources/beta-002'
        )
        is url_reference
    )


def test_collection_lookups_return_none_when_missing() -> None:
    collection = KnowledgeCollection()

    assert collection.get_record('missing-record') is None
    assert collection.get_reference_by_identifier('missing-id') is None
    assert (
        collection.get_reference_by_canonical_url(
            'https://example.test/missing'
        )
        is None
    )


@pytest.mark.parametrize(
    ('field_name', 'value', 'message'),
    [
        (
            'records',
            {'record': make_record('fictional-record-1')},
            'records must be an ordered sequence of CuratedKnowledgeRecord',
        ),
        (
            'references',
            {make_reference('one', identifier='fictional-id:one')},
            'references must be an ordered sequence of ScientificReference',
        ),
    ],
)
def test_collection_rejects_unordered_containers(
    field_name: str,
    value: object,
    message: str,
) -> None:
    arguments = {field_name: value}

    with pytest.raises(TypeError, match=rf'^{message}$'):
        KnowledgeCollection(**arguments)


@pytest.mark.parametrize(
    ('field_name', 'value', 'message'),
    [
        (
            'records',
            ['not-a-record'],
            r'records\[0\] must be a CuratedKnowledgeRecord',
        ),
        (
            'references',
            ['not-a-reference'],
            r'references\[0\] must be a ScientificReference',
        ),
    ],
)
def test_collection_validates_sequence_entries(
    field_name: str,
    value: object,
    message: str,
) -> None:
    arguments = {field_name: value}

    with pytest.raises(TypeError, match=rf'^{message}$'):
        KnowledgeCollection(**arguments)


def test_collection_rejects_duplicate_record_ids() -> None:
    first = make_record('fictional-record-1')
    second = make_record('fictional-record-1')

    with pytest.raises(
        ValueError,
        match=(
            r'^records\[1\]\.id duplicates records\[0\]\.id: '
            r"'fictional-record-1'$"
        ),
    ):
        KnowledgeCollection(records=[first, second])


def test_collection_rejects_duplicate_reference_identifiers() -> None:
    first = make_reference('one', identifier='fictional-id:shared')
    second = make_reference(
        'two',
        identifier='fictional-id:shared',
        canonical_url='https://example.test/sources/two',
    )

    with pytest.raises(
        ValueError,
        match=(
            r'^references\[1\]\.identifier duplicates '
            r'references\[0\]\.identifier: '
            r"'fictional-id:shared'$"
        ),
    ):
        KnowledgeCollection(references=[first, second])


def test_collection_rejects_duplicate_canonical_urls() -> None:
    shared_url = 'https://example.test/sources/shared'
    first = make_reference(
        'one',
        identifier='fictional-id:one',
        canonical_url=shared_url,
    )
    second = make_reference(
        'two',
        identifier='fictional-id:two',
        canonical_url=shared_url,
    )

    with pytest.raises(
        ValueError,
        match=(
            r'^references\[1\]\.canonical_url duplicates '
            r'references\[0\]\.canonical_url: '
            r"'https://example\.test/sources/shared'$"
        ),
    ):
        KnowledgeCollection(references=[first, second])


def test_collection_resolves_identifier_and_url_in_declared_order() -> None:
    identifier = 'fictional-id:alpha-001'
    canonical_url = 'https://example.test/sources/beta-002'
    identifier_reference = make_reference('identifier', identifier=identifier)
    url_reference = make_reference('URL', canonical_url=canonical_url)
    record = make_record(
        'fictional-record-1',
        references=[canonical_url, identifier],
    )
    collection = KnowledgeCollection(
        records=[record],
        references=[identifier_reference, url_reference],
    )

    assert collection.resolve_references(record.id) == (
        url_reference,
        identifier_reference,
    )


def test_collection_allows_shared_and_empty_record_references() -> None:
    shared_identifier = 'fictional-id:shared'
    shared_reference = make_reference('shared', identifier=shared_identifier)
    first_record = make_record(
        'fictional-record-1', references=[shared_identifier]
    )
    second_record = make_record(
        'fictional-record-2', references=[shared_identifier]
    )
    empty_record = make_record('fictional-record-empty')
    collection = KnowledgeCollection(
        records=[first_record, second_record, empty_record],
        references=[shared_reference],
    )

    assert collection.resolve_references(first_record.id) == (
        shared_reference,
    )
    assert collection.resolve_references(second_record.id) == (
        shared_reference,
    )
    assert collection.resolve_references(empty_record.id) == ()


@pytest.mark.parametrize(
    'reference_value',
    [
        'fictional-id:missing',
        'https://example.test/sources/missing',
    ],
    ids=['identifier', 'canonical-url'],
)
def test_collection_rejects_unresolved_record_references(
    reference_value: str,
) -> None:
    record = make_record('fictional-record-1', references=[reference_value])

    with pytest.raises(
        ValueError,
        match=(
            r"^record 'fictional-record-1' contains unresolved scientific "
            rf'reference {reference_value!r}$'
        ),
    ):
        KnowledgeCollection(records=[record])


def test_collection_rejects_ambiguous_cross_field_alias() -> None:
    shared_alias = 'https://example.test/sources/shared'
    identifier_reference = make_reference(
        'identifier', identifier=shared_alias
    )
    url_reference = make_reference('URL', canonical_url=shared_alias)

    with pytest.raises(
        ValueError,
        match=(
            r"^reference alias 'https://example\.test/sources/shared' is "
            r'ambiguous between references\[0\]\.identifier and '
            r'references\[1\]\.canonical_url$'
        ),
    ):
        KnowledgeCollection(references=[identifier_reference, url_reference])


def test_resolve_references_rejects_unknown_record() -> None:
    collection = KnowledgeCollection()

    with pytest.raises(
        KeyError,
        match=r'^"record not found: \'missing-record\'"$',
    ):
        collection.resolve_references('missing-record')
