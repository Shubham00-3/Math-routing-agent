export interface MathQuestion {
  question: string;
  subject?: QuestionType;
  difficulty_level?: number;
  context?: string;
}

export interface Step {
  step_number: number;
  description: string;
  explanation: string;
  formula?: string;
  visual_aid?: string;
}

export interface Solution {
  question: string;
  solution_id: string;
  steps: Step[];
  final_answer: string;
  confidence_score: number;
  source: 'knowledge_base' | 'web_search' | 'hybrid';
  subject: QuestionType;
  difficulty_level: number;
  processing_time: number;
  references?: string[];
  created_at: string;
}

export interface Feedback {
  solution_id: string;
  feedback_type: 'accuracy' | 'clarity' | 'completeness' | 'difficulty' | 'overall';
  rating: number;
  comments?: string;
  improvement_suggestions?: string;
  user_id?: string;
}

export interface FeedbackResponse {
  feedback_id: string;
  processed: boolean;
  improvements_applied: string[];
  next_suggestions: string[];
}

export type QuestionType = 
  | 'algebra' 
  | 'calculus' 
  | 'geometry' 
  | 'trigonometry' 
  | 'statistics' 
  | 'linear_algebra' 
  | 'discrete_math' 
  | 'number_theory';

export type FeedbackType = 'accuracy' | 'clarity' | 'completeness' | 'difficulty' | 'overall';

export interface WebSocketMessage {
  type: string;
  data?: any;
  timestamp: number;
}

export interface Analytics {
  knowledge_base: {
    total_entries: number;
    subject_distribution: Record<string, number>;
  };
  feedback: {
    total_feedback: number;
    average_rating: number;
    improvement_areas: string[];
  };
  performance: {
    avg_response_time: number;
    success_rate: number;
    total_queries: number;
  };
  subjects?: Record<string, {
    count: number;
    avg_rating: number;
    avg_confidence: number;
    common_topics: string[];
  }>;
}
