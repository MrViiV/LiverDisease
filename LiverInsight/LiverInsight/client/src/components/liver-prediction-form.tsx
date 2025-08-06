import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation } from "@tanstack/react-query";
import { insertPredictionSchema, type InsertPrediction } from "@shared/schema";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { User, Heart, Leaf, Stethoscope, FlaskConical, Brain, ArrowRight, Calendar, Users, Weight, Wine, Cigarette, Dna, Activity, Droplets, TrendingUp, BarChart3 } from "lucide-react";

interface LiverPredictionFormProps {
  onResult: (result: any) => void;
  onLoadingChange: (loading: boolean) => void;
}

export default function LiverPredictionForm({ onResult, onLoadingChange }: LiverPredictionFormProps) {
  const { toast } = useToast();
  const [progress, setProgress] = useState(0);

  const form = useForm<InsertPrediction>({
    resolver: zodResolver(insertPredictionSchema),
    defaultValues: {
      age: undefined,
      gender: undefined,
      bmi: undefined,
      alcoholConsumption: 2,
      smoking: undefined,
      geneticRisk: undefined,
      physicalActivity: 5,
      diabetes: undefined,
      hypertension: undefined,
      liverFunctionScore: 75,
    },
  });

  const predictMutation = useMutation({
    mutationFn: async (data: InsertPrediction) => {
      const response = await apiRequest("POST", "/api/predict", data);
      return response.json();
    },
    onSuccess: (result) => {
      onResult(result);
      onLoadingChange(false);
      toast({
        title: "Prediction Complete",
        description: "Your liver disease risk assessment is ready.",
      });
    },
    onError: (error: any) => {
      onLoadingChange(false);
      toast({
        title: "Prediction Failed",
        description: error.message || "Something went wrong. Please try again.",
        variant: "destructive",
      });
    },
  });

  const onSubmit = (data: InsertPrediction) => {
    onLoadingChange(true);
    predictMutation.mutate(data);
  };

  // Calculate progress based on filled fields
  const watchedValues = form.watch();
  const requiredFields = ['age', 'gender', 'bmi', 'smoking', 'geneticRisk', 'diabetes', 'hypertension'];
  const filledRequiredFields = requiredFields.filter(field => {
    const value = watchedValues[field as keyof InsertPrediction];
    return value !== undefined && value !== null && value !== '';
  }).length;
  const calculatedProgress = (filledRequiredFields / requiredFields.length) * 100;
  
  if (calculatedProgress !== progress) {
    setProgress(calculatedProgress);
  }

  return (
    <div className="space-y-6">
      {/* Progress Card */}
      <Card className="medical-card">
        <CardContent className="pt-6">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-semibold text-gray-900 flex items-center">
              <svg className="h-5 w-5 text-blue-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
              </svg>
              Assessment Progress
            </h4>
            <span className="text-sm font-medium text-blue-600">{Math.round(progress)}%</span>
          </div>
          <Progress value={progress} className="mb-3" />
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-gray-600">Required fields</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
              <span className="text-gray-600">Recommended</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Form */}
      <Card className="medical-card">
        <CardHeader className="medical-gradient text-white">
          <CardTitle className="text-xl font-semibold flex items-center">
            <svg className="h-6 w-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
            </svg>
            Patient Assessment Form
          </CardTitle>
        </CardHeader>

        <CardContent className="p-6">
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8" data-testid="prediction-form">
              {/* Demographics Section */}
              <div className="space-y-6">
                <h4 className="text-lg font-semibold text-gray-900 flex items-center border-b border-gray-200 pb-3">
                  <User className="h-5 w-5 text-blue-500 mr-2" />
                  Demographics
                </h4>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <FormField
                    control={form.control}
                    name="age"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel className="flex items-center text-sm font-medium text-gray-700">
                          <Calendar className="h-4 w-4 text-gray-400 mr-2" />
                          Age (years)
                          <span className="ml-1 text-red-500">*</span>
                        </FormLabel>
                        <FormControl>
                          <Input
                            {...field}
                            type="number"
                            min={1}
                            max={100}
                            placeholder="Enter age"
                            className="medical-input"
                            data-testid="input-age"
                            onChange={(e) => field.onChange(e.target.value ? parseInt(e.target.value) : undefined)}
                          />
                        </FormControl>
                        <p className="text-xs text-gray-500">Valid range: 1-100 years</p>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="gender"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel className="flex items-center text-sm font-medium text-gray-700">
                          <Users className="h-4 w-4 text-gray-400 mr-2" />
                          Gender
                          <span className="ml-1 text-red-500">*</span>
                        </FormLabel>
                        <Select onValueChange={field.onChange} defaultValue={field.value} data-testid="select-gender">
                          <FormControl>
                            <SelectTrigger className="medical-select">
                              <SelectValue placeholder="Select gender" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            <SelectItem value="male">Male</SelectItem>
                            <SelectItem value="female">Female</SelectItem>
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
              </div>

              {/* Physical Health Section */}
              <div className="space-y-6">
                <h4 className="text-lg font-semibold text-gray-900 flex items-center border-b border-gray-200 pb-3">
                  <Heart className="h-5 w-5 text-blue-500 mr-2" />
                  Physical Health Metrics
                </h4>
                
                <FormField
                  control={form.control}
                  name="bmi"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="flex items-center text-sm font-medium text-gray-700">
                        <Weight className="h-4 w-4 text-gray-400 mr-2" />
                        Body Mass Index (BMI)
                        <span className="ml-1 text-red-500">*</span>
                      </FormLabel>
                      <FormControl>
                        <Input
                          {...field}
                          type="number"
                          min={10}
                          max={50}
                          step={0.01}
                          placeholder="Enter BMI (e.g., 24.5)"
                          className="medical-input"
                          data-testid="input-bmi"
                          onChange={(e) => field.onChange(e.target.value ? parseFloat(e.target.value) : undefined)}
                        />
                      </FormControl>
                      <p className="text-xs text-gray-500">Valid range: 10.00-50.00 kg/m²</p>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="physicalActivity"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="flex items-center text-sm font-medium text-gray-700">
                        <Activity className="h-4 w-4 text-gray-400 mr-2" />
                        Physical Activity (hours/week)
                      </FormLabel>
                      <div className="px-4 py-3 bg-gray-50 rounded-lg">
                        <FormControl>
                          <Slider
                            min={0}
                            max={20}
                            step={0.1}
                            value={[field.value]}
                            onValueChange={(value) => field.onChange(value[0])}
                            className="medical-slider"
                            data-testid="slider-physical-activity"
                          />
                        </FormControl>
                        <div className="flex justify-between items-center mt-2">
                          <span className="text-xs text-gray-500">0 hrs</span>
                          <span className="text-sm font-medium text-blue-600 bg-blue-100 px-2 py-1 rounded">
                            {field.value.toFixed(1)} hrs/week
                          </span>
                          <span className="text-xs text-gray-500">20 hrs</span>
                        </div>
                      </div>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              {/* Lifestyle Factors Section */}
              <div className="space-y-6">
                <h4 className="text-lg font-semibold text-gray-900 flex items-center border-b border-gray-200 pb-3">
                  <Leaf className="h-5 w-5 text-green-500 mr-2" />
                  Lifestyle Factors
                </h4>
                
                <FormField
                  control={form.control}
                  name="alcoholConsumption"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="flex items-center text-sm font-medium text-gray-700">
                        <Wine className="h-4 w-4 text-gray-400 mr-2" />
                        Alcohol Consumption (drinks/week)
                      </FormLabel>
                      <div className="px-4 py-3 bg-gray-50 rounded-lg">
                        <FormControl>
                          <Slider
                            min={0}
                            max={20}
                            step={0.1}
                            value={[field.value]}
                            onValueChange={(value) => field.onChange(value[0])}
                            className="medical-slider"
                            data-testid="slider-alcohol-consumption"
                          />
                        </FormControl>
                        <div className="flex justify-between items-center mt-2">
                          <span className="text-xs text-gray-500">0 drinks</span>
                          <span className="text-sm font-medium text-blue-600 bg-blue-100 px-2 py-1 rounded">
                            {field.value.toFixed(1)} drinks/week
                          </span>
                          <span className="text-xs text-gray-500">20 drinks</span>
                        </div>
                      </div>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="smoking"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="flex items-center text-sm font-medium text-gray-700">
                        <Cigarette className="h-4 w-4 text-gray-400 mr-2" />
                        Smoking Status
                        <span className="ml-1 text-red-500">*</span>
                      </FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value} data-testid="select-smoking">
                        <FormControl>
                          <SelectTrigger className="medical-select">
                            <SelectValue placeholder="Select smoking status" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="yes">Yes</SelectItem>
                          <SelectItem value="no">No</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              {/* Medical History Section */}
              <div className="space-y-6">
                <h4 className="text-lg font-semibold text-gray-900 flex items-center border-b border-gray-200 pb-3">
                  <Stethoscope className="h-5 w-5 text-blue-500 mr-2" />
                  Medical History
                </h4>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <FormField
                    control={form.control}
                    name="diabetes"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel className="flex items-center text-sm font-medium text-gray-700">
                          <Droplets className="h-4 w-4 text-gray-400 mr-2" />
                          Diabetes
                          <span className="ml-1 text-red-500">*</span>
                        </FormLabel>
                        <Select onValueChange={field.onChange} defaultValue={field.value} data-testid="select-diabetes">
                          <FormControl>
                            <SelectTrigger className="medical-select">
                              <SelectValue placeholder="Select status" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            <SelectItem value="yes">Yes</SelectItem>
                            <SelectItem value="no">No</SelectItem>
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="hypertension"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel className="flex items-center text-sm font-medium text-gray-700">
                          <TrendingUp className="h-4 w-4 text-gray-400 mr-2" />
                          Hypertension
                          <span className="ml-1 text-red-500">*</span>
                        </FormLabel>
                        <Select onValueChange={field.onChange} defaultValue={field.value} data-testid="select-hypertension">
                          <FormControl>
                            <SelectTrigger className="medical-select">
                              <SelectValue placeholder="Select status" />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            <SelectItem value="yes">Yes</SelectItem>
                            <SelectItem value="no">No</SelectItem>
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <FormField
                  control={form.control}
                  name="geneticRisk"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="flex items-center text-sm font-medium text-gray-700">
                        <Dna className="h-4 w-4 text-gray-400 mr-2" />
                        Genetic Risk Assessment
                        <span className="ml-1 text-red-500">*</span>
                      </FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value} data-testid="select-genetic-risk">
                        <FormControl>
                          <SelectTrigger className="medical-select">
                            <SelectValue placeholder="Select genetic risk level" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="low">Low</SelectItem>
                          <SelectItem value="medium">Medium</SelectItem>
                          <SelectItem value="high">High</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-gray-500">Based on family history and genetic predisposition</p>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              {/* Laboratory Results Section */}
              <div className="space-y-6">
                <h4 className="text-lg font-semibold text-gray-900 flex items-center border-b border-gray-200 pb-3">
                  <FlaskConical className="h-5 w-5 text-blue-500 mr-2" />
                  Laboratory Results
                </h4>
                
                <FormField
                  control={form.control}
                  name="liverFunctionScore"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="flex items-center text-sm font-medium text-gray-700">
                        <BarChart3 className="h-4 w-4 text-gray-400 mr-2" />
                        Liver Function Test Score
                        <span className="ml-1 text-red-500">*</span>
                      </FormLabel>
                      <div className="px-4 py-3 bg-gray-50 rounded-lg">
                        <FormControl>
                          <Slider
                            min={0}
                            max={100}
                            step={0.1}
                            value={[field.value]}
                            onValueChange={(value) => field.onChange(value[0])}
                            className="medical-slider"
                            data-testid="slider-liver-function"
                          />
                        </FormControl>
                        <div className="flex justify-between items-center mt-2">
                          <span className="text-xs text-gray-500">0 (Poor)</span>
                          <span className="text-sm font-medium text-blue-600 bg-blue-100 px-2 py-1 rounded">
                            {field.value.toFixed(1)} points
                          </span>
                          <span className="text-xs text-gray-500">100 (Excellent)</span>
                        </div>
                      </div>
                      <p className="text-xs text-gray-500">Composite score based on ALT, AST, ALP, and bilirubin levels</p>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              {/* Submit Button */}
              <div className="pt-6 border-t border-gray-200">
                <Button
                  type="submit"
                  disabled={predictMutation.isPending}
                  className="w-full medical-gradient medical-gradient-hover text-white font-semibold py-4 px-6 rounded-xl transition-all duration-200 transform hover:scale-[1.02] hover:shadow-lg focus:outline-none focus:ring-4 focus:ring-blue-300 flex items-center justify-center space-x-3"
                  data-testid="button-predict"
                >
                  <Brain className="h-5 w-5" />
                  <span>{predictMutation.isPending ? "Analyzing..." : "Analyze Liver Disease Risk"}</span>
                  <ArrowRight className="h-5 w-5" />
                </Button>
                <p className="text-xs text-gray-500 text-center mt-3">Processing typically takes 2-3 seconds</p>
              </div>
            </form>
          </Form>
        </CardContent>
      </Card>
    </div>
  );
}
