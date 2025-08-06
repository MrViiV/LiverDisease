import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { CheckCircle, AlertTriangle, XCircle, Download, UserCheck, Lightbulb, TrendingUp, ThumbsUp, Info } from "lucide-react";

interface ResultsPanelProps {
  result: any;
  isLoading: boolean;
}

export default function ResultsPanel({ result, isLoading }: ResultsPanelProps) {
  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel?.toLowerCase()) {
      case 'low':
        return <CheckCircle className="text-green-600 h-8 w-8" />;
      case 'medium':
        return <AlertTriangle className="text-yellow-600 h-8 w-8" />;
      case 'high':
        return <XCircle className="text-red-600 h-8 w-8" />;
      default:
        return <CheckCircle className="text-green-600 h-8 w-8" />;
    }
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel?.toLowerCase()) {
      case 'low':
        return 'from-green-500 to-green-600';
      case 'medium':
        return 'from-yellow-500 to-yellow-600';
      case 'high':
        return 'from-red-500 to-red-600';
      default:
        return 'from-green-500 to-green-600';
    }
  };

  const getRiskBgColor = (riskLevel: string) => {
    switch (riskLevel?.toLowerCase()) {
      case 'low':
        return 'bg-green-100';
      case 'medium':
        return 'bg-yellow-100';
      case 'high':
        return 'bg-red-100';
      default:
        return 'bg-green-100';
    }
  };

  return (
    <Card className={`medical-card overflow-hidden ${!result && !isLoading ? 'opacity-50' : ''}`}>
      <CardHeader className="bg-gradient-to-r from-green-500 to-green-600 text-white">
        <CardTitle className="text-xl font-semibold flex items-center">
          <svg className="h-6 w-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Risk Assessment Results
        </CardTitle>
      </CardHeader>
      
      <CardContent className="p-6 space-y-6">
        {/* Loading State */}
        {isLoading && (
          <div className="text-center py-8" data-testid="loading-state">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-center text-gray-600">Analyzing your health data...</p>
          </div>
        )}

        {/* Results Display */}
        {result && !isLoading && (
          <div className="space-y-6 animate-fade-in" data-testid="results-display">
            {/* Risk Level Indicator */}
            <div className="text-center">
              <div className={`inline-flex items-center justify-center w-20 h-20 ${getRiskBgColor(result.riskLevel)} rounded-full mb-4`}>
                {getRiskIcon(result.riskLevel)}
              </div>
              <h5 className="text-2xl font-bold text-gray-900 mb-2" data-testid="text-risk-level">
                {result.riskLevel?.charAt(0).toUpperCase() + result.riskLevel?.slice(1)} Risk
              </h5>
              <p className="text-gray-600" data-testid="text-risk-description">
                {result.description || `Your assessment indicates a ${result.riskLevel} risk for liver disease based on current health factors.`}
              </p>
            </div>

            {/* Risk Score */}
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">Risk Score</span>
                <span className="text-sm font-bold text-blue-600" data-testid="text-risk-score">
                  {Math.round(result.riskScore)}/100
                </span>
              </div>
              <Progress 
                value={result.riskScore} 
                className={`h-3`}
                data-testid="progress-risk-score"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Low</span>
                <span>High</span>
              </div>
            </div>

            {/* Key Risk Factors */}
            <div className="space-y-3">
              <h6 className="font-semibold text-gray-900 flex items-center">
                <AlertTriangle className="h-4 w-4 text-amber-500 mr-2" />
                Key Factors Identified
              </h6>
              <div className="space-y-2" data-testid="risk-factors">
                {result.factors?.positive?.map((factor: string, index: number) => (
                  <div key={index} className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
                    <ThumbsUp className="h-4 w-4 text-green-600 flex-shrink-0" />
                    <span className="text-sm text-green-800">{factor}</span>
                  </div>
                ))}
                {result.factors?.concerning?.map((factor: string, index: number) => (
                  <div key={index} className="flex items-center space-x-3 p-3 bg-amber-50 rounded-lg">
                    <Info className="h-4 w-4 text-amber-600 flex-shrink-0" />
                    <span className="text-sm text-amber-800">{factor}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Recommendations */}
            <div className="bg-blue-50 rounded-lg p-4">
              <h6 className="font-semibold text-blue-900 mb-3 flex items-center">
                <Lightbulb className="h-4 w-4 text-blue-600 mr-2" />
                Recommendations
              </h6>
              <ul className="space-y-2 text-sm text-blue-800" data-testid="recommendations">
                {result.recommendations?.map((recommendation: string, index: number) => (
                  <li key={index} className="flex items-start space-x-2">
                    <CheckCircle className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
                    <span>{recommendation}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Next Steps */}
            <div className="border-t border-gray-200 pt-4">
              <h6 className="font-semibold text-gray-900 mb-3">Next Steps</h6>
              <div className="space-y-3">
                <Button 
                  variant="outline" 
                  className="w-full bg-blue-100 hover:bg-blue-200 text-blue-800 font-medium py-3 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center space-x-2"
                  data-testid="button-download"
                >
                  <Download className="h-4 w-4" />
                  <span>Download Report</span>
                </Button>
                <Button 
                  variant="outline" 
                  className="w-full bg-green-100 hover:bg-green-200 text-green-800 font-medium py-3 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center space-x-2"
                  data-testid="button-consultation"
                >
                  <UserCheck className="h-4 w-4" />
                  <span>Schedule Consultation</span>
                </Button>
              </div>
            </div>
          </div>
        )}

        {/* Default State */}
        {!result && !isLoading && (
          <div className="text-center py-8 text-gray-500" data-testid="default-state">
            <TrendingUp className="h-12 w-12 mx-auto mb-4 text-gray-300" />
            <p>Complete the assessment form to view your results</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
