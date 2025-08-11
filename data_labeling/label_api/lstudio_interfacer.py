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

    def import_task(self, tasks):
        self.project.import_tasks(tasks)
    
    def count_new_tasks(self, project_id):
        response = self.ls.get(f"/api/projects/{project_id}/tasks?status=new&limit=0")
        return response["meta"]["total"]
    
    def get_tasks(self):
        
        completed_tasks = self.project.get_tasks(
            filter_="status=completed",
            with_annotations=True
        )

        return completed_tasks
    

    def create_webhook(self, endpoint):
        header = {
            "url": endpoint,
            "Authorization": f"Token {os.getenv('LABEL_STUDIO_API_KEY')}",
            "Content-Type": "application/json"
        }