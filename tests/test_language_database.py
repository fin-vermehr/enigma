import pytest
from pytest import fixture

from nlp_takehome.src.language_database import LanguageDatabase, START_SEQUENCE_TOKEN, PAD_TOKEN, END_SEQUENCE_TOKEN, \
    PADDING_INDEX


@fixture()
def language_database():
    return LanguageDatabase('test_database')


def test_initialization_of_language_database(language_database):
    assert language_database.name == 'test_database'

    tokens = [PAD_TOKEN, START_SEQUENCE_TOKEN, END_SEQUENCE_TOKEN]
    assert language_database._items == tokens

    # check whether tokens map to their own index
    for index in range(len(tokens)):
        assert language_database._items[tokens[index]] == index
        assert language_database._inverse_items[index] == tokens[index]


def test_number_of_items_updates(language_database):
    assert language_database.number_of_items == 3
    language_database.add_record('R')
    assert language_database.number_of_items == 4


def test_get_item(language_database):
    assert language_database.get_item(PAD_TOKEN) == PADDING_INDEX


def test_get_index():
    assert language_database.get_index(PADDING_INDEX) == PAD_TOKEN


@pytest.mark.parametrize("record", ['a', '1', '['])
def test_add_record(record):
    language_database_two = LanguageDatabase('testDatabase')
    language_database_two.add_record(record)
    index_of_new_item = language_database_two.number_of_items - 1

    assert record in language_database_two._items
    assert language_database_two._inverse_items[index_of_new_item] == record
