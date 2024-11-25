import base64
import time
from typing import List
import requests

class GettyImagesAPI:
    def __init__(self, api_key, access_token, base_url:str=None):
        self.base_url = base_url or "https://api.gettyimages.com/v3/ai/image-generations"
        self.headers = {
            "accept": "application/json",
            "Api-Key": api_key,
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
    
    def initiate_image_generation(self, prompt:str, **kwargs):
      """
      Initiates the image generation request.
      
      Args:
        * prompt (str): text prompt for image generation
        * kwargs: other kwargs (except 'prompt') of getty image api, see here for details https://api.gettyimages.com/swagger/index.html#/AiGenerator/post_v3_ai_image_generations
      """
      getty_kwargs = {
        "prompt": prompt,
        **kwargs
      }
      response = requests.post(self.base_url, headers=self.headers, json=getty_kwargs)
      response.raise_for_status()
      return response

    def poll_generation_status(self, generation_request_id: str):
      """
      Polls the generation request until it completes or errors out.
      """
      poll_url = f"{self.base_url}/{generation_request_id}"
      while True:
        response = requests.get(poll_url, headers=self.headers)
        if response.status_code == 200:
          return response
        elif response.status_code == 202:
          time.sleep(1.0 + 1e-2)  # Poll every 1 second
        else:
          self.handle_error(response)

    def download_image_from_urls(self, urls:List[str]) -> List:
      images = []
      for url in urls:
        image_response = requests.get(url)
        if image_response.status_code == 200:
            image_base64 = base64.b64encode(image_response.content).decode('utf-8')
            images.append(image_base64)
        else:
            self.handle_error(image_response)
      return images
    
    def download_image_from_req_id(self, generation_request_id: str, download_urls: list):
        """
        Fetches and downloads images as base64-encoded strings for generated images.
        """
        image_base64_list = []
        for i, _ in enumerate(download_urls):
            while True:
                download_url = f"{self.base_url}/{generation_request_id}/images/{i}/download-sizes"
                response = requests.get(download_url, headers=self.headers)
                if response.status_code == 200:
                    url = response.json().get("url")
                    if url:
                        # Download the actual image
                        image_response = requests.get(url)
                        if image_response.status_code == 200:
                            # Convert image content to base64
                            image_base64 = base64.b64encode(image_response.content).decode('utf-8')
                            image_base64_list.append(image_base64)
                        else:
                            self.handle_error(image_response)
                    break  # Exit polling for this image once the URL is available
                
                elif response.status_code == 202:
                    time.sleep(1)  # Wait 1 second before retrying
                
                else:
                    self.handle_error(response)

        return image_base64_list

    @staticmethod
    def handle_error(response):
        """
        Handles unexpected errors during API calls.
        """
        error_message = response.json().get("message", "Unknown error occurred.")
        raise ValueError(f"Status code: {response.status_code}, message: {error_message}")

    def generate_images(self, prompt, **kwargs):
        """
        Orchestrates the image generation process.
        
        Args:
          * prompt (str): text prompt for image generation
          * kwargs: other kwargs (except 'prompt') of getty image api, see here for details https://api.gettyimages.com/swagger/index.html#/AiGenerator/post_v3_ai_image_generations
        """
        # Step 1: Initiate generation
        response = self.initiate_image_generation(prompt, **kwargs)
        
        if response.status_code in (200, 202):
          response_data = response.json()
          generation_request_id = response_data.get("generation_request_id")
          if not generation_request_id:
              raise ValueError("No generation request ID found in response.")

          if response.status_code == 202:
              response_data = self.poll_generation_status(generation_request_id).json()

          results = response_data.get("results", [])
          if not results:
              raise ValueError("No results found in the response.")

          urls = [each["url"] for each in results]
          return self.download_image_from_urls(urls)

        self.handle_error(response)

# Example usage:
# api = GettyImagesAPI("<YOUR_API_KEY>", "<YOUR_ACCESS_TOKEN>")
# images = api.generate_images("Prompt Text", "4:3")
# print(images)
