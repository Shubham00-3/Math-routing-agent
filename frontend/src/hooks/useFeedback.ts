import React, { useState, useCallback } from 'react';
import { useMutation, useQueryClient } from 'react-query';
import { feedbackAPI } from '../services/api';
import { websocketService } from '../services/websocket';
import { Feedback, FeedbackResponse } from '../types';
import toast from 'react-hot-toast';

interface UseFeedbackResult {
  isSubmitting: boolean;
  submitFeedback: (feedback: Feedback) => Promise<void>;
  lastResponse: FeedbackResponse | null;
}

export const useFeedback = (): UseFeedbackResult => {
  const [lastResponse, setLastResponse] = useState<FeedbackResponse | null>(null);
  const queryClient = useQueryClient();

  React.useEffect(() => {
    websocketService.onMessage('feedback_processed', (data) => {
      setLastResponse(data);
      toast.success('Feedback processed! Thank you for helping improve our solutions.');
      
      if (data.improvements_applied.length > 0) {
        toast.success(`Improvements applied: ${data.improvements_applied.join(', ')}`);
      }
    });

    websocketService.onMessage('feedback_received', (data) => {
      toast.loading('Processing your feedback...', { id: 'feedback-processing' });
    });

    return () => {
      websocketService.offMessage('feedback_processed');
      websocketService.offMessage('feedback_received');
    };
  }, []);

  const submitMutation = useMutation(
    (feedback: Feedback) => feedbackAPI.submitFeedback(feedback),
    {
      onSuccess: (data) => {
        setLastResponse(data);
        toast.success('Feedback submitted successfully!');
        
        queryClient.invalidateQueries('feedbackAnalytics');
        queryClient.invalidateQueries(['solutionFeedback', data.feedback_id]);
      },
      onError: (error: any) => {
        const errorMessage = error.response?.data?.detail || 'Failed to submit feedback';
        toast.error(errorMessage);
      },
    }
  );

  const submitFeedback = useCallback(
    async (feedback: Feedback) => {
      websocketService.sendMessage('solution_feedback', { feedback });
      
      await submitMutation.mutateAsync(feedback);
    },
    [submitMutation]
  );

  return {
    isSubmitting: submitMutation.isLoading,
    submitFeedback,
    lastResponse,
  };
};
