import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Calculator, Send, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { MathQuestion } from '../types';

const mathQuestionSchema = z.object({
  question: z.string().min(5, 'Question must be at least 5 characters'),
  subject: z.string().optional(),
  difficulty_level: z.number().min(1).max(10).optional(),
});

interface MathInputProps {
  onSubmit: (question: MathQuestion) => void;
  isLoading: boolean;
  className?: string;
}

export const MathInput: React.FC<MathInputProps> = ({ onSubmit, isLoading, className = '' }) => {
  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
  } = useForm<MathQuestion>({
    resolver: zodResolver(mathQuestionSchema),
    defaultValues: {
      difficulty_level: 5,
    },
  });

  const questionValue = watch('question');

  const onFormSubmit = (data: MathQuestion) => {
    onSubmit(data);
  };

  const exampleQuestions = [
    "Solve the quadratic equation x² + 5x + 6 = 0",
    "Find the derivative of f(x) = 3x² + 2x - 1",
    "Calculate the area of a circle with radius 5 units",
  ];

  return (
    <div className={`bg-white rounded-xl shadow-lg p-6 ${className}`}>
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-blue-100 rounded-lg">
          <Calculator className="h-6 w-6 text-blue-600" />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Ask a Math Question</h2>
          <p className="text-sm text-gray-600">Get step-by-step solutions with explanations</p>
        </div>
      </div>

      <form onSubmit={handleSubmit(onFormSubmit)} className="space-y-4">
        <div>
          <label htmlFor="question" className="form-label">
            Math Question *
          </label>
          <textarea
            {...register('question')}
            id="question"
            rows={3}
            className="form-input"
            placeholder="Enter your mathematical question here..."
            disabled={isLoading}
          />
          {errors.question && (
            <p className="mt-1 text-sm text-red-600">{errors.question.message}</p>
          )}
        </div>

        {!questionValue && (
          <div className="bg-gray-50 rounded-lg p-4">
            <p className="text-sm font-medium text-gray-700 mb-2">Try these examples:</p>
            <div className="space-y-2">
              {exampleQuestions.map((example, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => {
                    const event = { target: { name: 'question', value: example } };
                    register('question').onChange(event);
                  }}
                  className="block text-left text-sm text-blue-600 hover:text-blue-800 hover:bg-blue-50 p-2 rounded transition-colors w-full"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}

        <motion.button
          type="submit"
          disabled={isLoading || !questionValue}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {isLoading ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin" />
              Solving Problem...
            </>
          ) : (
            <>
              <Send className="h-5 w-5" />
              Solve Problem
            </>
          )}
        </motion.button>
      </form>
    </div>
  );
};
