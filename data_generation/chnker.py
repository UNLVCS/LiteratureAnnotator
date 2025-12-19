import json
from collections import defaultdict

class Chunker():
    
    """
        File must be a valid json file following the given structure
            {
                PMID: {
                    "Title": [],
                    "Abstract": [],
                    "Introduction": [],
                    "Methodology": [],
                    etc...
                }
            }
        
        overlap_size determines the overlap that will be created between chunks
    """
    
    def __init__(self):
        # pass
        self.article = []              #  Raw article as a list between sections
        self.article_id = 123456            #  PMID of article
        self.chunked_instances = []       #  Article as a chunked list

        # self.article = list(article_dict.values())
        # self.article_id = article_dict.keys[0]

    def set_chunk(self, article_dict: dict, overlap_size: int = 50):

        # self.article                #  Raw article as a list between sections
        # self.article_id             #  PMID of article
        # self.chunked_instances = []       #  Article as a chunked list

        # self.article = list(article_dict.values())
        # self.article_id = article_dict.keys[0]
        self.article_id, self.article = next(iter(article_dict.items()))
        self.article_title = self.article['Title']
        self.article = list(self.article.values())
        
        self.split_document()

    def get_chunked_article(self):
        chunked_doc = {"id": self.article_id,
                       "title": self.article_title,
                       "chunks": self.chunked_instances
                    }
        return chunked_doc
    
    """
        Will split article sections into valid chunks. If sections are greater than xxx length
            each section will be chunked itself while maintaining some overlap
    """

    def split_document(self):

        # to_chunk = ""
        to_chunk = []
        to_chnk_txt_size = 0
        for section in self.article:                # If section > 150 words then chunk that section itself
            # Some sections (e.g., Title, References) are plain strings while others are lists of strings.
            # Joining a string with '\n'.join(section) will insert newlines between EACH CHARACTER,
            if isinstance(section, list):
                section = '\n'.join(section)
            elif not isinstance(section, str):
                section = str(section)
            if to_chnk_txt_size > 150:
                to_chunk.append(section)
                # to_chunk += section
                self.overlap_window_chunker(to_chunk)
                to_chunk = []
                to_chnk_txt_size = 0
            else:
                # to_chunk += section
                to_chnk_txt_size += len(section)
                to_chunk.append(section)
        
        self.overlap_window_chunker(to_chunk)      # Catch and chunk remaining text that has not been chunked

    def overlap_window_chunker(self, text, overlap_size=50):

        for i in range(1, len(text)):

            # Normalize possible list inputs to strings; if already string, use as-is
            prev_item = text[i-1]
            next_item = text[i]
            if isinstance(prev_item, list):
                text_first = "".join(prev_item)
            else:
                text_first = str(prev_item)
            if isinstance(next_item, list):
                text_later = "".join(next_item)
            else:
                text_later = str(next_item)
            
            chunk_first = ""
            chunk_latter = ""
            
            if len(text_first) < overlap_size:
                chunk_latter = text_first + ". " + text_later
            else:
                chunk_latter = text_first[overlap_size:] + ". " +  text_later


            if len(text_later) < overlap_size:
                chunk_first = text_first + ". " +  text_later
            else:
                chunk_first = text_first + ". " +  text_later[:overlap_size]

            self.chunked_instances.append(chunk_first)
            self.chunked_instances.append(chunk_latter)
