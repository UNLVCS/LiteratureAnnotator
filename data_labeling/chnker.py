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
        self.article = list(self.article.values())
        self.split_document()

    def get_chunked_article(self):
        chunked_doc = {"id": self.article_id,
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
            section = '\n'.join(section)
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

            text_first = "".join(text[i-1])
            text_later = "".join(text[i])
            
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
