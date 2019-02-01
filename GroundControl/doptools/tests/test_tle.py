import pytest
import requests

import doptrack
import doptrack.tle as doptle


class TestSpacetrackQuery:

    def test_post_request_was_given_correct_input_with_given_payload(self, mocker):
        mocker.patch('doptrack.config.Config')
        mocker.patch('requests.Session.post')
        mocker.patch('requests.Session.get')

        doptle.spacetrack_query('some/query', username='test_username', password='test_password')
        requests.Session.post.assert_called_once_with(
                'https://www.space-track.org/ajaxauth/login',
                data={'identity': 'test_username', 'password': 'test_password'})

    def test_post_request_was_given_correct_input_with_config_payload(self, mocker):
        mocker.patch.object(
                doptrack.config.Config,
                'credentials',
                create=True,
                return_value={'space-track.org': {
                        'username': 'test_username',
                        'password': 'test_password'}})
        mocker.patch('requests.Session.post')
        mocker.patch('requests.Session.get')

        doptle.spacetrack_query('some/query', username='test_username', password='test_password')
        requests.Session.post.assert_called_once_with(
                'https://www.space-track.org/ajaxauth/login',
                data={'identity': 'test_username', 'password': 'test_password'})

    def test_get_request_was_given_correct_input(self, mocker):
        mocker.patch('doptrack.config.Config')
        mock_response = mocker.Mock()
        mock_status_code = mocker.PropertyMock(return_value=200)
        type(mock_response).status_code = mock_status_code
        mocker.patch('requests.Session.post', return_value=mock_response)
        mocker.patch('requests.Session.get')

        query = 'some/kind/of/query'
        full_query = 'https://www.space-track.org/basicspacedata/query/' + query
        doptle.spacetrack_query(query, username='test_username', password='test_password')
        requests.Session.get.assert_called_once_with(full_query)
