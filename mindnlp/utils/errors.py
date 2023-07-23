# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
MindNLP defined Errors.
"""
from typing import Optional
from requests import HTTPError, Response
from requests import JSONDecodeError

class MSHTTPError(HTTPError):
    """
    HTTPError to inherit from for any custom HTTP Error raised in MindNLP.

    Any HTTPError is converted at least into a `MSHTTPError`. If some information is
    sent back by the server, it will be added to the error message.

    Added details:
    - Request id from "X-Request-Id" header if exists.
    - Server error message from the header "X-Error-Message".
    - Server error message if we can found one in the response body.
    """

    request_id: Optional[str] = None
    server_message: Optional[str] = None

    def __init__(self, message: str, response: Optional[Response] = None):
        # Parse server information if any.
        if response is not None:
            self.request_id = response.headers.get("X-Request-Id")
            try:
                server_data = response.json()
            except JSONDecodeError:
                server_data = {}

            # Retrieve server error message from multiple sources
            server_message_from_headers = response.headers.get("X-Error-Message")
            server_message_from_body = server_data.get("error")
            server_multiple_messages_from_body = "\n".join(
                error["message"] for error in server_data.get("errors", []) if "message" in error
            )

            # Concatenate error messages
            _server_message = ""
            if server_message_from_headers is not None:  # from headers
                _server_message += server_message_from_headers + "\n"
            if server_message_from_body is not None:  # from body "error"
                if isinstance(server_message_from_body, list):
                    server_message_from_body = "\n".join(server_message_from_body)
                if server_message_from_body not in _server_message:
                    _server_message += server_message_from_body + "\n"
            if server_multiple_messages_from_body is not None:  # from body "errors"
                if server_multiple_messages_from_body not in _server_message:
                    _server_message += server_multiple_messages_from_body + "\n"
            _server_message = _server_message.strip()

            # Set message to `MSHTTPError` (if any)
            if _server_message != "":
                self.server_message = _server_message

        super().__init__(
            _format_error_message(
                message,
                request_id=self.request_id,
                server_message=self.server_message,
            ),
            response=response,
        )

    def append_to_message(self, additional_message: str) -> None:
        """Append additional information to the `HfHubHTTPError` initial message."""
        self.args = (self.args[0] + additional_message,) + self.args[1:]


class ModelNotFoundError(MSHTTPError):
    """
    Raised when trying to access a hf.co URL with an invalid repository name, or
    with a private repo name the user does not have access to.

    Example:

    ```py
    >>> from huggingface_hub import model_info
    >>> model_info("<non_existent_repository>")
    (...)
    huggingface_hub.utils._errors.RepositoryNotFoundError: 401 Client Error. (Request ID: PvMw_VjBMjVdMz53WKIzP)

    Repository Not Found for url: https://huggingface.co/api/models/%3Cnon_existent_repository%3E.
    Please make sure you specified the correct `repo_id` and `repo_type`.
    If the repo is private, make sure you are authenticated.
    Invalid username or password.
    ```
    """

def _format_error_message(message: str, request_id: Optional[str], server_message: Optional[str]) -> str:
    """
    Format the `HfHubHTTPError` error message based on initial message and information
    returned by the server.

    Used when initializing `HfHubHTTPError`.
    """
    # Add message from response body
    if server_message is not None and len(server_message) > 0 and server_message.lower() not in message.lower():
        if "\n\n" in message:
            message += "\n" + server_message
        else:
            message += "\n\n" + server_message

    # Add Request ID
    if request_id is not None and str(request_id).lower() not in message.lower():
        request_id_message = f" (Request ID: {request_id})"
        if "\n" in message:
            newline_index = message.index("\n")
            message = message[:newline_index] + request_id_message + message[newline_index:]
        else:
            message += request_id_message

    return message
