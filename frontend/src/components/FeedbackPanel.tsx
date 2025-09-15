import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Star, ThumbsUp, ThumbsDown, CheckCircle, Send, Loader2 } from 'lucide-react';
import { Feedback } from '../types';

interface FeedbackPanelProps {
  solutionId: string;
  onSubmit: (feedback: Feedback) => void;
  isSubmitting: boolean;
  className?: string;
}

export const FeedbackPanel: React.FC<FeedbackPanelProps> = ({
  solutionId,
  onSubmit,
  isSubmitting,
  className = '',
}) => {
  const [quickRating, setQuickRating] = useState<'good' | 'bad' | null>(null);
  const [showDetailed, setShowDetailed] = useState(false);
  const [rating, setRating] = useState(5);
  const [comments, setComments] = useState('');

  const handleQuickFeedback = (type: 'good' | 'bad') => {
    setQuickRating(type);
    
    const feedback: Feedback = {
      solution_id: solutionId,
      feedback_type: 'overall',
      rating: type === 'good' ? 5 : 2,
      comments: type === 'good' ? 'Helpful solution!' : 'Needs improvement',
    };
    
    onSubmit(feedback);
  };

  const handleDetailedSubmit = () => {
    const feedback: Feedback = {
      solution_id: solutionId,
      feedback_type: 'overall',
      rating,
      comments: comments.trim() || undefined,
    };
    
    onSubmit(feedback);
    setShowDetailed(false);
    setComments('');
    setRating(5);
  };

  if (quickRating && !showDetailed) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className={`bg-green-50 border border-green-200 rounded-lg p-4 ${className}`}
      >
        <div className="flex items-center gap-2 text-green-700">
          <CheckCircle className="h-5 w-5" />
          <span className="font-medium">Thank you for your feedback!</span>
        </div>
        <p className="text-sm text-green-600 mt-1">
          Your input helps us improve our solutions.
        </p>
      </motion.div>
    );
  }

  if (showDetailed) {
    return (
      <motion.div
        initial={{ opacity: 0, height: 0 }}
        animate={{ opacity: 1, height: 'auto' }}
        className={`bg-white rounded-lg shadow-md p-6 ${className}`}
      >
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Feedback</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Rating
            </label>
            <div className="flex items-center gap-1">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  type="button"
                  onClick={() => setRating(star)}
                  className="transition-colors"
                >
                  <Star
                    className={`h-5 w-5 ${
                      star <= rating
                        ? 'text-yellow-400 fill-current'
                        : 'text-gray-300 hover:text-yellow-300'
                    }`}
                  />
                </button>
              ))}
              <span className="ml-2 text-sm text-gray-600">
                {rating}/5
              </span>
            </div>
          </div>

          <div>
            <label htmlFor="comments" className="block text-sm font-medium text-gray-700 mb-2">
              Comments (Optional)
            </label>
            <textarea
              id="comments"
              value={comments}
              onChange={(e) => setComments(e.target.value)}
              rows={3}
              className="form-input"
              placeholder="What did you think about this solution?"
            />
          </div>

          <div className="flex justify-between items-center pt-4 border-t border-gray-200">
            <button
              onClick={() => setShowDetailed(false)}
              className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-700 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleDetailedSubmit}
              disabled={isSubmitting}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium rounded-lg transition-colors flex items-center gap-2"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Submitting...
                </>
              ) : (
                <>
                  <Send className="h-4 w-4" />
                  Submit Feedback
                </>
              )}
            </button>
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`bg-white rounded-lg shadow-md p-4 ${className}`}
    >
      <div className="text-center">
        <h3 className="text-lg font-medium text-gray-900 mb-2">Was this solution helpful?</h3>
        <div className="flex justify-center gap-4">
          <button
            onClick={() => handleQuickFeedback('good')}
            disabled={isSubmitting}
            className="flex items-center gap-2 px-4 py-2 bg-green-100 hover:bg-green-200 text-green-700 rounded-lg transition-colors disabled:opacity-50"
          >
            <ThumbsUp className="h-4 w-4" />
            Yes, helpful
          </button>
          <button
            onClick={() => handleQuickFeedback('bad')}
            disabled={isSubmitting}
            className="flex items-center gap-2 px-4 py-2 bg-red-100 hover:bg-red-200 text-red-700 rounded-lg transition-colors disabled:opacity-50"
          >
            <ThumbsDown className="h-4 w-4" />
            Needs work
          </button>
        </div>
        <button
          onClick={() => setShowDetailed(true)}
          className="mt-3 text-sm text-blue-600 hover:text-blue-700 underline"
        >
          Give detailed feedback
        </button>
      </div>
    </motion.div>
  );
};
