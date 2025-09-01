from label_studio_sdk import Client
import os
from dotenv import load_dotenv
import requests

class Labeller:
    def __init__(self):
        load_dotenv()
        self.ls = Client(url=os.getenv('LABEL_STUDIO_URL'), api_key=os.getenv('LABEL_STUDIO_API_KEY'))


        # Load XML label config
        with open('label_api/label_interface_config.xml', 'r') as f:
            interface_config = f.read()

        # Create a new project with your custom labeling interface
        self.project = self.ls.start_project(
            title='RAG Annotation Project',
            description='Labeling sections relevant to questions using a RAG pipeline',
            label_config=interface_config
        )

        # Headers for creating Webhooks later
        self.headers = {
            "Authorization": f"Token {os.getenv('LABEL_STUDIO_API_KEY')}"
        }
        

    def create_task(self, article):
        tasks = []
        for query in article['queries']:
            task = {
                "data": {
                    "paper_id": article['paper_id'],
                    "title": article['title'],
                    "paper_text": article[query]["retrieved_chunks"],
                    "class_criteria": query
                }
            }
            tasks.append(task)
        return tasks  # Return the created tasks

    def import_task(self, tasks):
        print("IMPORTED TASKS RESPONSE")
        print(self.project.import_tasks(tasks))
    
    def count_new_tasks(self, project_id):
        # Use requests directly since the SDK Client doesn't have a generic get method
        response = requests.get(
            f"{self.ls.url}/api/projects/{project_id}/tasks?status=new&limit=0",
            # f"{self.ls.url}/api/projects/{project_id}/tasks?status=new&limit=0",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()["meta"]["total"]
    
    def get_tasks(self):
        
        completed_tasks = self.project.get_tasks(
            filter_="status=completed",
            with_annotations=True
        )

        return completed_tasks
    

    def create_webhook(self, endpoint):
        """Create a webhook for the project with proper error handling and validation."""
        # Validate endpoint URL
        if not endpoint.startswith(('http://', 'https://')):
            raise ValueError("Endpoint URL must start with http:// or https://")
        
        if not endpoint.endswith('/webhook'):
            raise ValueError("Endpoint URL must end with /webhook for FastAPI handler")

        # Add Content-Type to headers if not present
        webhook_headers = self.headers.copy()
        webhook_headers.update({
            "Content-Type": "application/json"
        })

        try:
            response = requests.post(
                f"{self.ls.url}/api/webhooks/",
                headers=webhook_headers,
                json={
                    "url": endpoint,
                    "events": ["PROJECT_CREATED", "ANNOTATION_CREATED", "ANNOTATION_DELETED", "ANNOTATION_COMPLETED"],
                    "project": self.project.id,
                    "enabled": True,
                    "send_payload": True  # Ensure full payload is sent
                }
            )
            response.raise_for_status()  # Raise exception for bad status codes
            
            # Verify webhook was created
            webhook_data = response.json()
            if not webhook_data.get('id'):
                raise ValueError("Webhook creation failed - no webhook ID returned")
                
        except requests.exceptions.RequestException as e:
            print(f"Failed to create webhook: {str(e)}")
            raise
        print(f"Webhook created: {response.status_code} - {response.json()}")