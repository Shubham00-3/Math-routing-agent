import React, { useEffect, useState } from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';
import { motion } from 'framer-motion';
import { Calculator, History } from 'lucide-react';

import { MathInput } from '../components/MathInput';
import { SolutionDisplay } from '../components/SolutionDisplay';
import { FeedbackPanel } from '../components/FeedbackPanel';
import { useMathQuery } from '../hooks/useMathQuery';
import { useFeedback } from '../hooks/useFeedback';
import { websocketService } from '../services/websocket';
import { MathQuestion } from '../types';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

const MathApp: React.FC = () => {
  const {
    solution,
    isLoading,
    error,
    solveProblem,
    regenerateSolution,
    history,
  } = useMathQuery();

  const { submitFeedback, isSubmitting } = useFeedback();

  // Initialize WebSocket connection
  useEffect(() => {
    websocketService.connect();
    // No need to set isConnected state anymore

    return () => {
      websocketService.disconnect();
    };
  }, []);

  const handleQuestionSubmit = async (question: MathQuestion) => {
    try {
      await solveProblem(question);
    } catch (error) {
      console.error('Error solving problem:', error);
    }
  };

  const handleFeedbackSubmit = async (feedback: any) => {
    try {
      await submitFeedback(feedback);
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  const handleRegenerate = async (solutionId: string) => {
    try {
      await regenerateSolution(solutionId);
    } catch (error) {
      console.error('Error regenerating solution:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Calculator className="h-6 w-6 text-blue-600" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Math Routing Agent</h1>
                <p className="text-xs text-gray-500">
                  AI-powered mathematical tutoring with human feedback learning
                </p>
              </div>
            </div>
            {/* The Connection Status has been removed from here */}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          {/* Math Input */}
          <MathInput
            onSubmit={handleQuestionSubmit}
            isLoading={isLoading}
          />

          {/* Error Display */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-red-50 border border-red-200 rounded-lg p-4"
            >
              <p className="text-red-800">{error}</p>
            </motion.div>
          )}

          {/* Solution Display */}
          {solution && (
            <div className="space-y-6">
              <SolutionDisplay
                solution={solution}
                isRegenerating={isLoading}
              />
              
              {/* Feedback Panel */}
              <FeedbackPanel
                solutionId={solution.solution_id}
                onSubmit={handleFeedbackSubmit}
                isSubmitting={isSubmitting}
              />
            </div>
          )}

          {/* Loading State */}
          {isLoading && !solution && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="bg-white rounded-xl shadow-lg p-8"
            >
              <div className="flex items-center justify-center space-x-3">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <span className="text-gray-600 font-medium">Generating solution...</span>
              </div>
              <div className="mt-4 space-y-2">
                <div className="h-2 bg-gray-200 rounded animate-pulse"></div>
                <div className="h-2 bg-gray-200 rounded animate-pulse w-3/4"></div>
                <div className="h-2 bg-gray-200 rounded animate-pulse w-1/2"></div>
              </div>
            </motion.div>
          )}

          {/* History Section */}
          {history.length > 0 && (
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center gap-2 mb-4">
                <History className="h-5 w-5 text-gray-600" />
                <h2 className="text-lg font-semibold text-gray-900">Recent Solutions</h2>
              </div>
              <div className="space-y-3">
                {history.slice(0, 3).map((historySolution) => (
                  <div
                    key={historySolution.solution_id}
                    className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer"
                  >
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {historySolution.question}
                    </p>
                    <div className="flex items-center gap-4 mt-1 text-xs text-gray-600">
                      <span className="capitalize">{historySolution.subject.replace('_', ' ')}</span>
                      <span>{(historySolution.confidence_score * 100).toFixed(0)}% confident</span>
                      <span>{new Date(historySolution.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Toast Notifications */}
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
          success: {
            style: {
              background: '#059669',
            },
          },
          error: {
            style: {
              background: '#DC2626',
            },
          },
        }}
      />
    </div>
  );
};

export default function Home() {
  return (
    <QueryClientProvider client={queryClient}>
      <MathApp />
    </QueryClientProvider>
  );
}