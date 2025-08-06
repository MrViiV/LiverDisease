import { Stethoscope, Shield, Award, Heart } from "lucide-react";
import LiverPredictionForm from "@/components/liver-prediction-form";
import ResultsPanel from "@/components/results-panel";
import { useState } from "react";

export default function Home() {
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-blue-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-blue-600 p-3 rounded-xl">
                <Heart className="text-white h-6 w-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Liver Disease Prediction</h1>
                <p className="text-sm text-gray-600">AI-Powered Medical Assessment Tool</p>
              </div>
            </div>
            <div className="hidden md:flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <Stethoscope className="h-4 w-4 text-blue-500" />
                <span>Clinically Validated</span>
              </div>
            </div>
          </div>
        </div>
      </header>
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Introduction Section */}
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Liver Health Risk Assessment</h2>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Complete the comprehensive health questionnaire below to receive an AI-powered assessment of your liver disease risk factors. 
            This tool is designed to assist healthcare professionals in preliminary screening.
          </p>
          <div className="mt-6 bg-amber-50 border border-amber-200 rounded-lg p-4 max-w-2xl mx-auto">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-amber-500 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="text-sm text-amber-800">
                <strong>Medical Disclaimer:</strong> This tool is for educational and screening purposes only. Always consult with a qualified healthcare professional for proper medical diagnosis and treatment.
              </div>
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Assessment Form */}
          <div className="lg:col-span-2">
            <LiverPredictionForm 
              onResult={setPredictionResult} 
              onLoadingChange={setIsLoading}
            />
          </div>

          {/* Results Panel */}
          <div className="space-y-6">
            <ResultsPanel 
              result={predictionResult} 
              isLoading={isLoading}
            />

            {/* Information Panel */}
            <div className="medical-card p-6">
              <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <svg className="h-5 w-5 text-blue-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
                About This Assessment
              </h4>
              <div className="space-y-4 text-sm text-gray-600">
                <p>This AI-powered tool analyzes multiple risk factors to provide a preliminary assessment of liver disease risk. The algorithm considers:</p>
                <ul className="space-y-1 ml-4">
                  <li>• Demographic factors (age, gender)</li>
                  <li>• Lifestyle indicators (diet, exercise, substances)</li>
                  <li>• Medical history and comorbidities</li>
                  <li>• Laboratory test results</li>
                  <li>• Genetic predisposition factors</li>
                </ul>

              </div>
            </div>
          </div>
        </div>
      </main>
      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-6 mb-4">
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <Shield className="h-4 w-4 text-green-500" />
                <span>Data Protected</span>
              </div>
              
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <Stethoscope className="h-4 w-4 text-green-500" />
                <span>Clinically Validated</span>
              </div>
            </div>
            <p className="text-sm text-gray-500">© 2025 Liver Health Analytics. For assessment purposes use only.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
