# import uuid
# import logging
# from typing import List, Optional, Dict, Tuple
# from langchain_huggingface import HuggingFaceEmbeddings  # Removed incorrect import
# from qdrant_client import QdrantClient
# from qdrant_client.models import (
#     VectorParams, Distance, PointStruct, Filter, FieldCondition, 
#     MatchValue, SearchRequest, CollectionInfo
# )
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import numpy as np
# from datetime import datetime
# import json

# from app.models.schemas import (
#     KnowledgeEntry, QuestionType, SolutionResponse, Step
# )
# from app.core.config import settings

# logger = logging.getLogger(__name__)

# class KnowledgeBaseService:
#     """Vector database service for mathematical knowledge base"""
    
#     def __init__(self):
#         self.client = QdrantClient(
#             host=settings.QDRANT_HOST,
#             port=settings.QDRANT_PORT
#         )
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="all-MiniLM-L6-v2",  # Fast, good quality, free
#             model_kwargs={'device': 'cpu'},  # Use CPU (or 'cuda' if you have GPU)
#             encode_kwargs={'normalize_embeddings': True}
#         )
#         self.collection_name = settings.QDRANT_COLLECTION_NAME
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
#         )
        
#     async def initialize_collection(self):
#         """Initialize Qdrant collection with proper configuration"""
#         try:
#             # Check if collection exists
#             collections = self.client.get_collections()
#             collection_names = [col.name for col in collections.collections]
            
#             if self.collection_name not in collection_names:
#                 logger.info(f"Creating collection: {self.collection_name}")
#                 self.client.create_collection(
#                     collection_name=self.collection_name,
#                     vectors_config=VectorParams(
#                         size=384,  # Changed from 1536 to 384 for all-MiniLM-L6-v2
#                         distance=Distance.COSINE
#                     )
#                 )
#                 logger.info("Collection created successfully")
#             else:
#                 logger.info(f"Collection {self.collection_name} already exists")
                
#         except Exception as e:
#             logger.error(f"Error initializing collection: {str(e)}")
#             raise

#     async def add_knowledge_entry(self, entry: KnowledgeEntry) -> str:
#         """Add a knowledge entry to the vector database"""
#         try:
#             # Create searchable text from question and solution
#             searchable_text = self._create_searchable_text(entry)
            
#             # Generate embedding
#             embedding = await self._generate_embedding(searchable_text)
            
#             # Prepare metadata
#             metadata = {
#                 "question": entry.question,
#                 "subject": entry.subject.value,
#                 "difficulty": entry.difficulty,
#                 "tags": entry.tags,
#                 "created_at": entry.last_accessed.isoformat(),
#                 "usage_count": entry.usage_count,
#                 "solution_id": entry.id,
#                 "final_answer": entry.solution.final_answer,
#                 "step_count": len(entry.solution.steps),
#                 "confidence_score": entry.solution.confidence_score
#             }
            
#             # Create point
#             point = PointStruct(
#                 id=entry.id,
#                 vector=embedding,
#                 payload=metadata
#             )
            
#             # Insert into Qdrant
#             self.client.upsert(
#                 collection_name=self.collection_name,
#                 points=[point]
#             )
            
#             logger.info(f"Added knowledge entry: {entry.id}")
#             return entry.id
            
#         except Exception as e:
#             logger.error(f"Error adding knowledge entry: {str(e)}")
#             raise

#     async def search_similar_questions(
#         self, 
#         question: str, 
#         subject: Optional[QuestionType] = None,
#         difficulty_range: Optional[Tuple[int, int]] = None,
#         limit: int = 5,
#         score_threshold: float = 0.7
#     ) -> List[Tuple[KnowledgeEntry, float]]:
#         """Search for similar questions in the knowledge base"""
#         try:
#             # Generate query embedding
#             query_embedding = await self._generate_embedding(question)
            
#             # Prepare filters
#             filters = []
#             if subject:
#                 filters.append(
#                     FieldCondition(
#                         key="subject",
#                         match=MatchValue(value=subject.value)
#                     )
#                 )
            
#             if difficulty_range:
#                 min_diff, max_diff = difficulty_range
#                 filters.append(
#                     FieldCondition(
#                         key="difficulty",
#                         range={
#                             "gte": min_diff,
#                             "lte": max_diff
#                         }
#                     )
#                 )
            
#             # Perform vector search
#             search_result = self.client.search(
#                 collection_name=self.collection_name,
#                 query_vector=query_embedding,
#                 query_filter=Filter(must=filters) if filters else None,
#                 limit=limit,
#                 score_threshold=score_threshold
#             )
            
#             # Convert results to KnowledgeEntry objects
#             results = []
#             for point in search_result:
#                 try:
#                     knowledge_entry = await self._point_to_knowledge_entry(point)
#                     results.append((knowledge_entry, point.score))
#                 except Exception as e:
#                     logger.warning(f"Error converting point to knowledge entry: {str(e)}")
#                     continue
            
#             # Update usage count for accessed entries
#             for entry, _ in results:
#                 await self._update_usage_count(entry.id)
            
#             logger.info(f"Found {len(results)} similar questions for query: {question[:50]}...")
#             return results
            
#         except Exception as e:
#             logger.error(f"Error searching similar questions: {str(e)}")
#             return []

#     async def get_knowledge_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
#         """Retrieve a specific knowledge entry by ID"""
#         try:
#             result = self.client.retrieve(
#                 collection_name=self.collection_name,
#                 ids=[entry_id],
#                 with_payload=True,
#                 with_vectors=True
#             )
            
#             if result:
#                 point = result[0]
#                 return await self._point_to_knowledge_entry(point)
            
#             return None
            
#         except Exception as e:
#             logger.error(f"Error retrieving knowledge entry {entry_id}: {str(e)}")
#             return None

#     async def update_knowledge_entry(self, entry: KnowledgeEntry) -> bool:
#         """Update an existing knowledge entry"""
#         try:
#             # Check if entry exists
#             existing = await self.get_knowledge_entry(entry.id)
#             if not existing:
#                 logger.warning(f"Knowledge entry {entry.id} not found for update")
#                 return False
            
#             # Add updated entry (upsert)
#             await self.add_knowledge_entry(entry)
#             return True
            
#         except Exception as e:
#             logger.error(f"Error updating knowledge entry: {str(e)}")
#             return False

#     async def delete_knowledge_entry(self, entry_id: str) -> bool:
#         """Delete a knowledge entry"""
#         try:
#             self.client.delete(
#                 collection_name=self.collection_name,
#                 points_selector=[entry_id]
#             )
#             logger.info(f"Deleted knowledge entry: {entry_id}")
#             return True
            
#         except Exception as e:
#             logger.error(f"Error deleting knowledge entry: {str(e)}")
#             return False

#     async def get_collection_stats(self) -> Dict:
#         """Get collection statistics"""
#         try:
#             info = self.client.get_collection(self.collection_name)
            
#             # Get subject distribution
#             subject_stats = {}
#             for subject in QuestionType:
#                 count_result = self.client.count(
#                     collection_name=self.collection_name,
#                     count_filter=Filter(
#                         must=[
#                             FieldCondition(
#                                 key="subject",
#                                 match=MatchValue(value=subject.value)
#                             )
#                         ]
#                     )
#                 )
#                 subject_stats[subject.value] = count_result.count
            
#             return {
#                 "total_entries": info.points_count,
#                 "vector_size": info.config.params.vectors.size,
#                 "distance_metric": info.config.params.vectors.distance,
#                 "subject_distribution": subject_stats,
#                 "indexed": info.status == "green"
#             }
            
#         except Exception as e:
#             logger.error(f"Error getting collection stats: {str(e)}")
#             return {}

#     # Private helper methods
#     def _create_searchable_text(self, entry: KnowledgeEntry) -> str:
#         """Create searchable text from knowledge entry"""
#         # Combine question, solution steps, and answer
#         text_parts = [entry.question]
        
#         # Add solution steps
#         for step in entry.solution.steps:
#             text_parts.append(f"Step {step.step_number}: {step.description}")
#             if step.explanation:
#                 text_parts.append(step.explanation)
#             if step.formula:
#                 text_parts.append(f"Formula: {step.formula}")
        
#         # Add final answer
#         text_parts.append(f"Answer: {entry.solution.final_answer}")
        
#         # Add tags and subject
#         text_parts.extend(entry.tags)
#         text_parts.append(entry.subject.value)
        
#         return " ".join(text_parts)

#     async def _generate_embedding(self, text: str) -> List[float]:
#         """Generate embedding for text"""
#         try:
#             embedding = self.embeddings.embed_query(text)
#             return embedding
#         except Exception as e:
#             logger.error(f"Error generating embedding: {str(e)}")
#             # Return zero vector as fallback
#             return [0.0] * 384

#     async def _point_to_knowledge_entry(self, point) -> KnowledgeEntry:
#         """Convert Qdrant point to KnowledgeEntry object"""
#         payload = point.payload
        
#         # Reconstruct solution (this would typically be stored in a separate DB)
#         # For now, we'll create a minimal solution object
#         solution = SolutionResponse(
#             question=payload["question"],
#             solution_id=payload["solution_id"],
#             steps=[],  # Steps would be reconstructed from separate storage
#             final_answer=payload["final_answer"],
#             confidence_score=payload["confidence_score"],
#             source="knowledge_base",
#             subject=QuestionType(payload["subject"]),
#             difficulty_level=payload["difficulty"],
#             processing_time=0.0,
#             created_at=datetime.fromisoformat(payload["created_at"])
#         )
        
#         return KnowledgeEntry(
#             id=str(point.id),
#             question=payload["question"],
#             solution=solution,
#             tags=payload["tags"],
#             difficulty=payload["difficulty"],
#             subject=QuestionType(payload["subject"]),
#             embedding=point.vector,
#             usage_count=payload["usage_count"],
#             last_accessed=datetime.fromisoformat(payload["created_at"])
#         )

#     async def _update_usage_count(self, entry_id: str) -> None:
#         """Update usage count for a knowledge entry"""
#         try:
#             # Get current entry
#             result = self.client.retrieve(
#                 collection_name=self.collection_name,
#                 ids=[entry_id],
#                 with_payload=True
#             )
            
#             if result:
#                 current_payload = result[0].payload
#                 current_payload["usage_count"] = current_payload.get("usage_count", 0) + 1
#                 current_payload["last_accessed"] = datetime.utcnow().isoformat()
                
#                 # Update the point
#                 self.client.set_payload(
#                     collection_name=self.collection_name,
#                     payload=current_payload,
#                     points=[entry_id]
#                 )
                
#         except Exception as e:
#             logger.warning(f"Error updating usage count for {entry_id}: {str(e)}")

# backend/app/services/knowledge_base.py
# import uuid
# import logging
# from typing import List, Optional, Dict, Tuple
# from qdrant_client import QdrantClient
# from qdrant_client.models import (
#     VectorParams, Distance, PointStruct, Filter, FieldCondition,
#     MatchValue, SearchRequest, CollectionInfo
# )
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import numpy as np
# from datetime import datetime
# import json

# from app.models.schemas import (
#     KnowledgeEntry, QuestionType, SolutionResponse, Step
# )
# from app.core.config import settings

# logger = logging.getLogger(__name__)

# class KnowledgeBaseService:
#     """Vector database service for mathematical knowledge base"""

#     def __init__(self):
#         self.client = QdrantClient(
#             host=settings.QDRANT_HOST,
#             port=settings.QDRANT_PORT
#         )
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="all-MiniLM-L6-v2",
#             model_kwargs={'device': 'cpu'},
#             encode_kwargs={'normalize_embeddings': True}
#         )
#         self.collection_name = settings.QDRANT_COLLECTION_NAME
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
#         )

#     async def initialize_collection(self):
#         """Initialize Qdrant collection with proper configuration"""
#         try:
#             collections = self.client.get_collections()
#             collection_names = [col.name for col in collections.collections]

#             if self.collection_name not in collection_names:
#                 logger.info(f"Creating collection: {self.collection_name}")
#                 self.client.create_collection(
#                     collection_name=self.collection_name,
#                     vectors_config=VectorParams(
#                         size=384,  # Correct size for all-MiniLM-L6-v2
#                         distance=Distance.COSINE
#                     )
#                 )
#                 logger.info("Collection created successfully")
#             else:
#                 logger.info(f"Collection {self.collection_name} already exists")

#         except Exception as e:
#             logger.error(f"Error initializing collection: {str(e)}")
#             raise

#     async def add_knowledge_entry(self, entry: KnowledgeEntry) -> str:
#         """Add a knowledge entry to the vector database"""
#         try:
#             searchable_text = self._create_searchable_text(entry)
#             embedding = await self._generate_embedding(searchable_text)
            
#             metadata = {
#                 "question": entry.question,
#                 "subject": entry.subject.value,
#                 "difficulty": entry.difficulty,
#                 "tags": entry.tags,
#                 "created_at": entry.last_accessed.isoformat(),
#                 "usage_count": entry.usage_count,
#                 "solution_id": entry.id,
#                 "final_answer": entry.solution.final_answer,
#                 "step_count": len(entry.solution.steps),
#                 "confidence_score": entry.solution.confidence_score
#             }
            
#             point = PointStruct(
#                 id=entry.id,
#                 vector=embedding,
#                 payload=metadata
#             )
            
#             self.client.upsert(
#                 collection_name=self.collection_name,
#                 points=[point]
#             )
            
#             logger.info(f"Added knowledge entry: {entry.id}")
#             return entry.id
            
#         except Exception as e:
#             logger.error(f"Error adding knowledge entry: {str(e)}")
#             raise

#     async def search_similar_questions(
#         self,
#         question: str,
#         subject: Optional[QuestionType] = None,
#         difficulty_range: Optional[Tuple[int, int]] = None,
#         limit: int = 5,
#         score_threshold: float = 0.7
#     ) -> List[Tuple[KnowledgeEntry, float]]:
#         """Search for similar questions in the knowledge base"""
#         try:
#             query_embedding = await self._generate_embedding(question)
            
#             filters = []
#             if subject:
#                 filters.append(
#                     FieldCondition(
#                         key="subject",
#                         match=MatchValue(value=subject.value)
#                     )
#                 )
            
#             if difficulty_range:
#                 min_diff, max_diff = difficulty_range
#                 filters.append(
#                     FieldCondition(
#                         key="difficulty",
#                         range={
#                             "gte": min_diff,
#                             "lte": max_diff
#                         }
#                     )
#                 )
            
#             search_result = self.client.search(
#                 collection_name=self.collection_name,
#                 query_vector=query_embedding,
#                 query_filter=Filter(must=filters) if filters else None,
#                 limit=limit,
#                 score_threshold=score_threshold
#             )
            
#             results = []
#             for point in search_result:
#                 try:
#                     knowledge_entry = await self._point_to_knowledge_entry(point)
#                     results.append((knowledge_entry, point.score))
#                 except Exception as e:
#                     logger.warning(f"Error converting point to knowledge entry: {str(e)}")
#                     continue
            
#             for entry, _ in results:
#                 await self._update_usage_count(entry.id)
            
#             logger.info(f"Found {len(results)} similar questions for query: {question[:50]}...")
#             return results
            
#         except Exception as e:
#             logger.error(f"Error searching similar questions: {str(e)}")
#             return []

#     async def get_recent_entries(self, limit: int = 10, offset: int = 0) -> List[KnowledgeEntry]:
#         """Retrieve recent entries from the knowledge base"""
#         try:
#             response, _ = self.client.scroll(
#                 collection_name=self.collection_name,
#                 limit=limit,
#                 offset=offset,
#                 with_payload=True,
#                 with_vectors=False
#             )
            
#             entries = []
#             for point in response:
#                 try:
#                     entry = await self._point_to_knowledge_entry(point)
#                     entries.append(entry)
#                 except Exception as e:
#                     logger.warning(f"Error converting point to knowledge entry in history: {str(e)}")
            
#             return entries
#         except Exception as e:
#             logger.error(f"Error getting recent entries: {str(e)}")
#             return []


#     def _create_searchable_text(self, entry: KnowledgeEntry) -> str:
#         """Create searchable text from knowledge entry"""
#         text_parts = [entry.question]
        
#         for step in entry.solution.steps:
#             text_parts.append(f"Step {step.step_number}: {step.description}")
#             if step.explanation:
#                 text_parts.append(step.explanation)
#             if step.formula:
#                 text_parts.append(f"Formula: {step.formula}")
        
#         text_parts.append(f"Answer: {entry.solution.final_answer}")
        
#         text_parts.extend(entry.tags)
#         text_parts.append(entry.subject.value)
        
#         return " ".join(text_parts)

#     async def _generate_embedding(self, text: str) -> List[float]:
#         """Generate embedding for text"""
#         try:
#             embedding = self.embeddings.embed_query(text)
#             return embedding
#         except Exception as e:
#             logger.error(f"Error generating embedding: {str(e)}")
#             return [0.0] * 384

#     async def _point_to_knowledge_entry(self, point) -> KnowledgeEntry:
#         """Convert Qdrant point to KnowledgeEntry object"""
#         payload = point.payload
        
#         solution = SolutionResponse(
#             question=payload["question"],
#             solution_id=payload["solution_id"],
#             steps=[],
#             final_answer=payload["final_answer"],
#             confidence_score=payload["confidence_score"],
#             source=SourceType.KNOWLEDGE_BASE,
#             subject=QuestionType(payload["subject"]),
#             difficulty_level=payload["difficulty"],
#             processing_time=0.0,
#             created_at=datetime.fromisoformat(payload["created_at"])
#         )
        
#         return KnowledgeEntry(
#             id=str(point.id),
#             question=payload["question"],
#             solution=solution,
#             tags=payload["tags"],
#             difficulty=payload["difficulty"],
#             subject=QuestionType(payload["subject"]),
#             embedding=point.vector if hasattr(point, 'vector') and point.vector is not None else [],
#             usage_count=payload["usage_count"],
#             last_accessed=datetime.fromisoformat(payload["created_at"])
#         )
        
#     async def _update_usage_count(self, entry_id: str) -> None:
#         """Update usage count for a knowledge entry"""
#         try:
#             result = self.client.retrieve(
#                 collection_name=self.collection_name,
#                 ids=[entry_id],
#                 with_payload=True
#             )
            
#             if result:
#                 current_payload = result[0].payload
#                 current_payload["usage_count"] = current_payload.get("usage_count", 0) + 1
#                 current_payload["last_accessed"] = datetime.utcnow().isoformat()
                
#                 self.client.set_payload(
#                     collection_name=self.collection_name,
#                     payload=current_payload,
#                     points=[entry_id]
#                 )
                
#         except Exception as e:
#             logger.warning(f"Error updating usage count for {entry_id}: {str(e)}")
            
#     async def get_collection_stats(self) -> Dict:
#         """Get collection statistics"""
#         try:
#             info = self.client.get_collection(self.collection_name)
            
#             subject_stats = {}
#             for subject in QuestionType:
#                 count_result = self.client.count(
#                     collection_name=self.collection_name,
#                     count_filter=Filter(
#                         must=[
#                             FieldCondition(
#                                 key="subject",
#                                 match=MatchValue(value=subject.value)
#                             )
#                         ]
#                     ),
#                     exact=True
#                 )
#                 subject_stats[subject.value] = count_result.count
            
#             return {
#                 "total_entries": info.points_count,
#                 "vector_size": info.vectors_config.params.size,
#                 "distance_metric": info.vectors_config.params.distance,
#                 "subject_distribution": subject_stats,
#                 "indexed": info.status == "green"
#             }
            
#         except Exception as e:
#             logger.error(f"Error getting collection stats: {str(e)}")
#             return {}

# backend/app/services/knowledge_base.py
import uuid
import logging
from typing import List, Optional, Dict, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, FieldCondition,
    MatchValue, SearchRequest, CollectionInfo
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from datetime import datetime
import json

from app.models.schemas import (
    KnowledgeEntry, QuestionType, SolutionResponse, Step, SourceType
)
from app.core.config import settings

logger = logging.getLogger(__name__)

class KnowledgeBaseService:
    """Vector database service for mathematical knowledge base"""

    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    async def initialize_collection(self):
        """Initialize Qdrant collection with proper configuration"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise

    async def add_knowledge_entry(self, entry: KnowledgeEntry) -> str:
        """Add a knowledge entry to the vector database"""
        try:
            searchable_text = self._create_searchable_text(entry)
            embedding = await self._generate_embedding(searchable_text)
            
            metadata = {
                "question": entry.question,
                "subject": entry.subject.value,
                "difficulty": entry.difficulty,
                "tags": entry.tags,
                "created_at": entry.last_accessed.isoformat(),
                "usage_count": entry.usage_count,
                "solution_id": entry.id,
                "final_answer": entry.solution.final_answer,
                "step_count": len(entry.solution.steps),
                "confidence_score": entry.solution.confidence_score
            }
            
            point = PointStruct(
                id=entry.id,
                vector=embedding,
                payload=metadata
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Added knowledge entry: {entry.id}")
            return entry.id
            
        except Exception as e:
            logger.error(f"Error adding knowledge entry: {str(e)}")
            raise

    async def search_similar_questions(
        self,
        question: str,
        subject: Optional[QuestionType] = None,
        difficulty_range: Optional[Tuple[int, int]] = None,
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Tuple[KnowledgeEntry, float]]:
        """Search for similar questions in the knowledge base"""
        try:
            query_embedding = await self._generate_embedding(question)
            
            filters = []
            if subject:
                filters.append(
                    FieldCondition(
                        key="subject",
                        match=MatchValue(value=subject.value)
                    )
                )
            
            if difficulty_range:
                min_diff, max_diff = difficulty_range
                filters.append(
                    FieldCondition(
                        key="difficulty",
                        range={
                            "gte": min_diff,
                            "lte": max_diff
                        }
                    )
                )
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=Filter(must=filters) if filters else None,
                limit=limit,
                score_threshold=score_threshold
            )
            
            results = []
            for point in search_result:
                try:
                    knowledge_entry = await self._point_to_knowledge_entry(point)
                    results.append((knowledge_entry, point.score))
                except Exception as e:
                    logger.warning(f"Error converting point to knowledge entry: {str(e)}")
                    continue
            
            for entry, _ in results:
                await self._update_usage_count(entry.id)
            
            logger.info(f"Found {len(results)} similar questions for query: {question[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar questions: {str(e)}")
            return []
            
    async def get_recent_entries(self, limit: int = 10, offset: int = 0) -> List[KnowledgeEntry]:
        """Retrieve recent entries from the knowledge base"""
        try:
            response, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            entries = []
            for point in response:
                try:
                    entry = await self._point_to_knowledge_entry(point)
                    entries.append(entry)
                except Exception as e:
                    logger.warning(f"Error converting point to knowledge entry in history: {str(e)}")
            
            return entries
        except Exception as e:
            logger.error(f"Error getting recent entries: {str(e)}")
            return []

    def _create_searchable_text(self, entry: KnowledgeEntry) -> str:
        """Create searchable text from knowledge entry"""
        text_parts = [entry.question]
        
        for step in entry.solution.steps:
            text_parts.append(f"Step {step.step_number}: {step.description}")
            if step.explanation:
                text_parts.append(step.explanation)
            if step.formula:
                text_parts.append(f"Formula: {step.formula}")
        
        text_parts.append(f"Answer: {entry.solution.final_answer}")
        
        text_parts.extend(entry.tags)
        text_parts.append(entry.subject.value)
        
        return " ".join(text_parts)

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * 384

    async def _point_to_knowledge_entry(self, point) -> KnowledgeEntry:
        """Convert Qdrant point to KnowledgeEntry object"""
        payload = point.payload
        
        solution = SolutionResponse(
            question=payload["question"],
            solution_id=payload["solution_id"],
            steps=[],
            final_answer=payload["final_answer"],
            confidence_score=payload["confidence_score"],
            source=SourceType.KNOWLEDGE_BASE,
            subject=QuestionType(payload["subject"]),
            difficulty_level=payload["difficulty"],
            processing_time=0.0,
            created_at=datetime.fromisoformat(payload["created_at"])
        )
        
        return KnowledgeEntry(
            id=str(point.id),
            question=payload["question"],
            solution=solution,
            tags=payload["tags"],
            difficulty=payload["difficulty"],
            subject=QuestionType(payload["subject"]),
            embedding=point.vector if hasattr(point, 'vector') and point.vector is not None else [],
            usage_count=payload["usage_count"],
            last_accessed=datetime.fromisoformat(payload["created_at"])
        )
        
    async def _update_usage_count(self, entry_id: str) -> None:
        """Update usage count for a knowledge entry"""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[entry_id],
                with_payload=True
            )
            
            if result:
                current_payload = result[0].payload
                current_payload["usage_count"] = current_payload.get("usage_count", 0) + 1
                current_payload["last_accessed"] = datetime.utcnow().isoformat()
                
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload=current_payload,
                    points=[entry_id]
                )
                
        except Exception as e:
            logger.warning(f"Error updating usage count for {entry_id}: {str(e)}")
            
    async def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            
            subject_stats = {}
            for subject in QuestionType:
                count_result = self.client.count(
                    collection_name=self.collection_name,
                    count_filter=Filter(
                        must=[
                            FieldCondition(
                                key="subject",
                                match=MatchValue(value=subject.value)
                            )
                        ]
                    ),
                    exact=True
                )
                subject_stats[subject.value] = count_result.count
            
            return {
                "total_entries": info.points_count,
                "vector_size": info.vectors_config.params.size,
                "distance_metric": info.vectors_config.params.distance,
                "subject_distribution": subject_stats,
                "indexed": info.status == "green"
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}