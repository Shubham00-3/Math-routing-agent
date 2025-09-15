import React, { useState, useCallback } from 'react';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import { mathAPI } from '../services/api';
import { websocketService } from '../services/websocket';
import { MathQuestion, Solution } from '../types';
import toast from 'react-hot-toast';

interface UseMathQueryResult {
  solution: Solution | null;
  isLoading: boolean;
  error: string | null;
  solveProblem: (question: MathQuestion) => Promise<void>;
  regenerateSolution: (solutionId: string, context?: string) => Promise<void>;
  history: Solution[];
  isLoadingHistory: boolean;
}

export const useMathQuery = (): UseMathQueryResult => {
  const [solution, setSolution] = useState<Solution | null>(null);
  const [error, setError] = useState<string | null>(null);
  const queryClient = useQueryClient();

  React.useEffect(() => {
    websocketService.onMessage('solution_complete', (data) => {
      toast.success('Solution generated successfully!');
      queryClient.invalidateQueries('mathHistory');
    });

    websocketService.onMessage('solution_status', (data) => {
      if (data.status === 'processing') {
        toast.loading(data.message, { id: 'solution-processing' });
      } else if (data.status === 'complete') {
        toast.success(data.message, { id: 'solution-processing' });
      }
    });

    return () => {
      websocketService.offMessage('solution_complete');
      websocketService.offMessage('solution_status');
    };
  }, [queryClient]);

  const solveMutation = useMutation(
    (question: MathQuestion) => mathAPI.solveProblem(question),
    {
      onSuccess: (data) => {
        setSolution(data);
        setError(null);
        toast.success('Problem solved successfully!');
        queryClient.invalidateQueries('mathHistory');
      },
      onError: (error: any) => {
        const errorMessage = error.response?.data?.detail || 'Failed to solve problem';
        setError(errorMessage);
        toast.error(errorMessage);
      },
    }
  );

  const regenerateMutation = useMutation(
    ({ solutionId, context }: { solutionId: string; context?: string }) =>
      mathAPI.regenerateSolution(solutionId, context),
    {
      onSuccess: (data) => {
        setSolution(data);
        setError(null);
        toast.success('Solution regenerated with improvements!');
        queryClient.invalidateQueries('mathHistory');
      },
      onError: (error: any) => {
        const errorMessage = error.response?.data?.detail || 'Failed to regenerate solution';
        setError(errorMessage);
        toast.error(errorMessage);
      },
    }
  );

  const { data: history = [], isLoading: isLoadingHistory } = useQuery(
    'mathHistory',
    () => mathAPI.getSolutionHistory(20),
    {
      staleTime: 5 * 60 * 1000,
      cacheTime: 10 * 60 * 1000,
    }
  );

  const solveProblem = useCallback(
    async (question: MathQuestion) => {
      setError(null);
      setSolution(null);
      
      websocketService.sendMessage('solution_request', { question });
      
      await solveMutation.mutateAsync(question);
    },
    [solveMutation]
  );

  const regenerateSolution = useCallback(
    async (solutionId: string, context?: string) => {
      setError(null);
      await regenerateMutation.mutateAsync({ solutionId, context });
    },
    [regenerateMutation]
  );

  return {
    solution,
    isLoading: solveMutation.isLoading || regenerateMutation.isLoading,
    error,
    solveProblem,
    regenerateSolution,
    history,
    isLoadingHistory,
  };
};
