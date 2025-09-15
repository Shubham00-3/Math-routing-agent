import axios from 'axios';
import { MathQuestion, Solution, Feedback, FeedbackResponse, Analytics } from '../types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const mathAPI = {
  async solveProblem(question: MathQuestion): Promise<Solution> {
    const response = await api.post('/math/solve', question);
    return response.data;
  },

  async getSolution(solutionId: string): Promise<Solution> {
    const response = await api.get(`/math/solution/${solutionId}`);
    return response.data;
  },

  async getSolutionHistory(limit = 10, offset = 0): Promise<Solution[]> {
    const response = await api.get('/math/history', {
      params: { limit, offset },
    });
    return response.data;
  },

  async regenerateSolution(solutionId: string, feedbackContext?: string): Promise<Solution> {
    const response = await api.post(`/math/regenerate/${solutionId}`, {
      feedback_context: feedbackContext,
    });
    return response.data;
  },
};

export const feedbackAPI = {
  async submitFeedback(feedback: Feedback): Promise<FeedbackResponse> {
    const response = await api.post('/feedback/submit', feedback);
    return response.data;
  },

  async getSolutionFeedback(solutionId: string) {
    const response = await api.get(`/feedback/solution/${solutionId}`);
    return response.data;
  },

  async getFeedbackAnalytics(days = 7) {
    const response = await api.get('/feedback/analytics', {
      params: { days },
    });
    return response.data;
  },

  async triggerOptimization(force = false) {
    const response = await api.post('/feedback/optimize', { force });
    return response.data;
  },
};

export const analyticsAPI = {
  async getDashboard(days = 7): Promise<Analytics> {
    const response = await api.get('/analytics/dashboard', {
      params: { days },
    });
    return response.data;
  },

  async getPerformanceMetrics(days = 7) {
    const response = await api.get('/analytics/performance', {
      params: { days },
    });
    return response.data;
  },

  async getSubjectAnalytics(days = 7) {
    const response = await api.get('/analytics/subjects', {
      params: { days },
    });
    return response.data;
  },

  async getTrends(days = 30) {
    const response = await api.get('/analytics/trends', {
      params: { days },
    });
    return response.data;
  },
};

export const healthAPI = {
  async getHealth() {
    const response = await api.get('/health');
    return response.data;
  },

  async getDetailedHealth() {
    const response = await api.get('/health/detailed');
    return response.data;
  },
};
