from typing import List
import httpx
from urllib.parse import urljoin
from pydantic_core import ValidationError
import base64
from app.logger import logger
import os
import urllib
import json
from basyx.aas.model import AssetAdministrationShell, Submodel
from basyx.aas.adapter.json import AASToJsonEncoder

def encode_id(id: str):
    return base64.b64encode(bytes(id, 'utf-8')).decode('ascii')


def decode_id(id: str):
    id_dec = base64.b64decode(id).decode('ascii')
    return id_dec


class BasyxApiClient:


    def __init__(self, base_url, auth_token=None, headers={}):

        self.base_url = base_url
        self.auth_token = auth_token
        self.headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else headers
        self.client = httpx.AsyncClient(headers=self.headers)


    async def get_shell(self, aas_id: str):

        response = await self.get(f"/shells/{encode_id(aas_id)}")
        return response
    

    async def get_shells(self):

        response = await self.get("/shells")
        shells = response.get('result')
        # logger.info(f"shells on the server: {shells}")
        return shells
    

    async def add_shell(self, shell):

        try:
            _url = self.base_url + "/shells"
            # logger.info(f"Posting AAS '{shell.get('idShort')}' at '{_url}'")
            try:
                http_response = await self.post(_url, data=shell)
                logger.info(f"Response: {http_response}")
            except ValidationError:
                print("Validation Error happend during registry of AAS Descriptor, but registry process has been successful. Bug?")
        except Exception as e:
            print("Exception: "+str(e))
            print("Error during AAS registration")


    async def delete_shell(self, shell_id):
        
        _url = self.base_url + f"/shells/{shell_id}"
        logger.info(f"Deleting AAS '{shell_id}' at '{_url}'")
        try:
            http_response = await self.client.delete(_url)
            logger.info(f"Response: {http_response}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e}")
        except httpx.RequestError as e:
            logger.error(f"Request error occurred: {e}")
        except Exception as e:
            logger.error(f"An error occurred while deleting '{shell_id}': {e}")


    async def get_submodels(self, submodel_ids : List[str]):

        submodels = []
        for submodel_id in submodel_ids:
            try:
                response = await self.get_submodel(submodel_id)
                submodels.append(response)
            except Exception as e:
                logger.error(f"An error occurred while fetching submodels: {e}")
        return submodels


    async def get_submodel(self, submodel_id):

        _url = f"/submodels/{encode_id(submodel_id)}"
        logger.info(f"Getting Submodel '{submodel_id}' at '{_url}'")
        try:
            submodel = await self.get(_url)
        except Exception as e:
            logger.error(f"An error occurred while fetching submodel '{submodel_id}': {e}")
        return submodel


    async def add_submodel(self, submodel):

        try:
            _url = f"{self.base_url}/submodels"
            logger.info(f"Posting Submodel '{submodel.get('idShort')}' at '{_url}'")
            try:
                http_response = await self.post(_url, data=submodel)
                logger.info(f"Response: {http_response}")
            except ValidationError:
                print("Validation Error happend during registry of AAS Descriptor, but registry process has been successful. Bug?")
        except Exception as e:
            print("Exception: "+str(e))
            print("Error during AAS registration")


    async def append_submodel(self, submodel, aas_id):

        try:
            logger.info(f"Appending Submodel '{submodel.get('idShort')}' to AAS '{aas_id}'")
            try:
                _url = f"{self.base_url}/submodels"
                http_response = await self.post(_url, data=submodel)
                logger.info(f"Create Response: {http_response}")
                _url = f"{self.base_url}/shells/{aas_id}/submodel-refs"
                http_response = await self.post(_url, data={
                    "type": "ModelReference",
                    "keys": [{"value": submodel.get('id'), "type": "Submodel"}]
                })
                logger.info(f"Registering Response: {http_response}")
            except ValidationError:
                print("Validation Error happend during registry of AAS Descriptor, but registry process has been successful. Bug?")
        except Exception as e:
            print("Exception: "+str(e))
            print("Error during AAS registration")


    async def register_aas(self, aas: AssetAdministrationShell):
        logger.info("Registering AAS")
        try:
            aas_obj = json.loads(json.dumps(aas, cls=AASToJsonEncoder))
            await self.add_shell(aas_obj)
        except Exception as e:
            logger.error(f"Error during AAS registration: {e}")
            return False
        return True

    async def register_submodel(self, submodel: Submodel):
        logger.info("Registering submodels")
        try:
            # submodel_obj = json.loads(json.dumps(submodel, cls=AASToJsonEncoder))
            await self.add_submodel(submodel)
        except Exception as e:
            logger.error(f"Error during submodel registration: {e}")
            return False
        return True


    async def get(self, path, params=None):
        try:
            _url = urljoin(self.base_url, path)
            response = await self.client.get(_url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e}")
        except httpx.RequestError as e:
            logger.error(f"Request error occurred: {e}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    async def patch(self, path, data, reraise=False):
        try:
            _url = urljoin(self.base_url, path)
            response = await self.client.patch(_url, data=data)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e.response.status_code} --- {e.__class__.__name__}")
            return response
        except httpx.RequestError as e:
            logger.error(f"Request error occurred: {e}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")


    async def post(self, path, data, reraise=False):
        try:
            _url = urljoin(self.base_url, path)
            response = await self.client.post(_url, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e.response.status_code} --- {e.__class__.__name__}")
            return response
        except httpx.RequestError as e:
            logger.error(f"Request error occurred: {e}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    async def put(self, path, data, reraise=False):
        try:
            _url = urljoin(self.base_url, path)
            response = await self.client.put(_url, json=data)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {e.response.status_code} --- {e.__class__.__name__}")
            return response
        except httpx.RequestError as e:
            logger.error(f"Request error occurred: {e}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    async def download_aas_package(self, aas_id: str, submodel_ids: List[str], filepath: str):
        """Download AASX package for the given AAS and submodels."""
        try:
            b64_aas_id = urllib.parse.quote(base64.urlsafe_b64encode(aas_id.encode()).decode(), safe='')
            b64_submodel_ids = [
                urllib.parse.quote(base64.urlsafe_b64encode(sm_id.encode()).decode(), safe='')
                for sm_id in submodel_ids
            ]

            query_params = {
                "aasIds": b64_aas_id,
                "submodelIds": b64_submodel_ids,
                "includeConceptDescriptions": "true"
            }

            headers = {
                "Accept": "application/asset-administration-shell-package+xml"
            }

            response = await self.client.get(
                urljoin(self.base_url, "/serialization"),
                params=query_params,
                headers=headers
            )
            response.raise_for_status()

            # Write the AASX file
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(response.content)

            logger.info(f"AASX package downloaded to: {filepath}")
            return filepath

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred during download: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during AASX download: {e}")

    async def download_aas_json(self, aas_id: str, submodel_ids: List[str], filepath: str):
        """Download AAS as JSON for the given AAS and submodels."""
        try:
            b64_aas_id = urllib.parse.quote(base64.urlsafe_b64encode(aas_id.encode()).decode(), safe='')
            b64_submodel_ids = [
                urllib.parse.quote(base64.urlsafe_b64encode(sm_id.encode()).decode(), safe='')
                for sm_id in submodel_ids
            ]

            query_params = {
                "aasIds": b64_aas_id,
                "submodelIds": b64_submodel_ids,
                "includeConceptDescriptions": "true"
            }

            headers = {
                "Accept": "application/json"
            }

            response = await self.client.get(
                urljoin(self.base_url, "/serialization"),
                params=query_params,
                headers=headers
            )
            response.raise_for_status()

            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Write JSON to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(response.json(), f, indent=2)

            logger.info(f"AAS JSON downloaded to: {filepath}")
            return filepath

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred during JSON download: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during JSON download: {e}")


    async def register_submodels(self, submodel: Submodel):
        try:
            await self.client.add_submodel(submodel)
        except Exception as e:
            logger.error(f"Error during submodel registration: {e}")
            return False
        return True

# Example usage
async def main():
    api_client = BasyxApiClient("http://localhost:8081")
    response = await api_client.get_shells()
    print(response)



if __name__ == "__main__":
    import asyncio
    asyncio.run(main())