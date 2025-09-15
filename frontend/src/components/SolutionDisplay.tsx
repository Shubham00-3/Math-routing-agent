import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, Clock, Copy, RefreshCw } from 'lucide-react';
import toast from 'react-hot-toast';
import { Solution } from '../types';

interface SolutionDisplayProps {
  solution: Solution;
  onRegenerate?: (solutionId: string) => void;
  isRegenerating?: boolean;
  className?: string;
}

export const SolutionDisplay: React.FC<SolutionDisplayProps> = ({
  solution,
  onRegenerate,
  isRegenerating = false,
  className = '',
}) => {
  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success('Copied to clipboard!');
    } catch (error) {
      toast.error('Failed to copy');
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-100 text-green-800';
    if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`bg-white rounded-xl shadow-lg overflow-hidden ${className}`}
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 text-white p-6">
        <h2 className="text-xl font-semibold mb-2">Solution</h2>
        <p className="text-blue-100">{solution.question}</p>
        
        <div className="flex items-center gap-4 mt-4 text-sm">
          <div className="flex items-center gap-1">
            <Clock className="h-4 w-4" />
            {solution.processing_time.toFixed(2)}s
          </div>
          <div className={`px-2 py-1 rounded-full text-xs font-medium ${getConfidenceColor(solution.confidence_score)}`}>
            {(solution.confidence_score * 100).toFixed(0)}% confident
          </div>
          <div className="capitalize">
            {solution.subject.replace('_', ' ')}
          </div>
        </div>
      </div>

      {/* Solution Steps */}
      <div className="p-6">
        <div className="space-y-4">
          {solution.steps.map((step) => (
            <div key={step.step_number} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <span className="bg-blue-100 text-blue-600 text-sm font-medium px-2 py-1 rounded">
                  Step {step.step_number}
                </span>
                <div className="flex-1">
                  <div className="font-medium text-gray-900 mb-2">
                    {step.description}
                  </div>
                  {step.explanation && (
                    <div className="text-gray-600 text-sm">
                      {step.explanation}
                    </div>
                  )}
                  {step.formula && (
                    <div className="mt-2 bg-gray-50 p-2 rounded font-mono text-sm">
                      {step.formula}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Final Answer */}
        <div className="mt-8 p-6 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-200">
          <div className="flex items-start gap-3">
            <CheckCircle className="h-6 w-6 text-green-600 mt-1 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-green-900 mb-2">Final Answer</h3>
              <div className="text-green-800 text-lg font-medium">
                {solution.final_answer}
              </div>
              <button
                onClick={() => copyToClipboard(solution.final_answer)}
                className="mt-3 text-sm text-green-600 hover:text-green-700 flex items-center gap-1 transition-colors"
              >
                <Copy className="h-4 w-4" />
                Copy Answer
              </button>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-3 mt-6 pt-4 border-t border-gray-200">
          {onRegenerate && (
            <button
              onClick={() => onRegenerate(solution.solution_id)}
              disabled={isRegenerating}
              className="px-4 py-2 text-sm font-medium text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              <RefreshCw className={`h-4 w-4 ${isRegenerating ? 'animate-spin' : ''}`} />
              {isRegenerating ? 'Regenerating...' : 'Regenerate'}
            </button>
          )}
        </div>
      </div>
    </motion.div>
  );
};
