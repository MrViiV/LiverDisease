import { type User, type InsertUser, type Prediction, type InsertPrediction } from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  getUser(id: string): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  predictLiverDisease(data: InsertPrediction): Promise<Prediction>;
  savePrediction(prediction: Prediction): Promise<Prediction>;
  getAllPredictions(): Promise<Prediction[]>;
}

export class MemStorage implements IStorage {
  private users: Map<string, User>;
  private predictions: Map<string, Prediction>;

  constructor() {
    this.users = new Map();
    this.predictions = new Map();
  }

  async getUser(id: string): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = randomUUID();
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  async predictLiverDisease(data: InsertPrediction): Promise<Prediction> {
    // Liver disease prediction algorithm
    const riskScore = this.calculateRiskScore(data);
    const riskLevel = this.determineRiskLevel(riskScore);
    const factors = this.analyzeRiskFactors(data);
    const recommendations = this.generateRecommendations(data, riskLevel);

    const prediction: Prediction = {
      id: randomUUID(),
      ...data,
      riskScore,
      riskLevel,
      createdAt: new Date().toISOString(),
    };

    // Save prediction to memory
    await this.savePrediction(prediction);

    // Return enhanced result with additional analysis
    return {
      ...prediction,
      description: this.getRiskDescription(riskLevel),
      factors,
      recommendations,
    } as any;
  }

  async savePrediction(prediction: Prediction): Promise<Prediction> {
    this.predictions.set(prediction.id, prediction);
    return prediction;
  }

  async getAllPredictions(): Promise<Prediction[]> {
    return Array.from(this.predictions.values());
  }

  private calculateRiskScore(data: InsertPrediction): number {
    let score = 0;

    // Age factor (0-25 points)
    if (data.age > 60) score += 25;
    else if (data.age > 45) score += 15;
    else if (data.age > 30) score += 8;
    else score += 3;

    // Gender factor (0-5 points)
    if (data.gender === 'male') score += 5;

    // BMI factor (0-15 points)
    if (data.bmi > 30) score += 15;
    else if (data.bmi > 25) score += 8;
    else if (data.bmi < 18.5) score += 5;

    // Alcohol consumption (0-20 points)
    if (data.alcoholConsumption > 14) score += 20;
    else if (data.alcoholConsumption > 7) score += 12;
    else if (data.alcoholConsumption > 3) score += 6;

    // Smoking (0-10 points)
    if (data.smoking === 'yes') score += 10;

    // Genetic risk (0-15 points)
    if (data.geneticRisk === 'high') score += 15;
    else if (data.geneticRisk === 'medium') score += 8;
    else score += 2;

    // Physical activity (subtract points for good activity)
    if (data.physicalActivity > 10) score -= 8;
    else if (data.physicalActivity > 5) score -= 4;
    else if (data.physicalActivity < 1) score += 5;

    // Diabetes (0-10 points)
    if (data.diabetes === 'yes') score += 10;

    // Hypertension (0-8 points)
    if (data.hypertension === 'yes') score += 8;

    // Liver function score (0-25 points, inverse relationship)
    if (data.liverFunctionScore < 30) score += 25;
    else if (data.liverFunctionScore < 50) score += 15;
    else if (data.liverFunctionScore < 70) score += 8;
    else if (data.liverFunctionScore < 85) score += 3;

    // Ensure score is between 0 and 100
    return Math.max(0, Math.min(100, score));
  }

  private determineRiskLevel(riskScore: number): string {
    if (riskScore < 30) return 'low';
    else if (riskScore < 60) return 'medium';
    else return 'high';
  }

  private getRiskDescription(riskLevel: string): string {
    switch (riskLevel) {
      case 'low':
        return 'Your assessment indicates a low risk for liver disease based on current health factors.';
      case 'medium':
        return 'Your assessment shows moderate risk factors that should be monitored and addressed.';
      case 'high':
        return 'Your assessment indicates elevated risk factors that require immediate medical attention.';
      default:
        return 'Risk assessment completed based on provided health information.';
    }
  }

  private analyzeRiskFactors(data: InsertPrediction) {
    const positive: string[] = [];
    const concerning: string[] = [];

    // Analyze positive factors
    if (data.bmi >= 18.5 && data.bmi <= 25) positive.push('Normal BMI range');
    if (data.alcoholConsumption <= 3) positive.push('Low alcohol consumption');
    if (data.smoking === 'no') positive.push('Non-smoker status');
    if (data.physicalActivity >= 5) positive.push('Regular physical activity');
    if (data.diabetes === 'no') positive.push('No diabetes');
    if (data.hypertension === 'no') positive.push('No hypertension');
    if (data.liverFunctionScore >= 75) positive.push('Good liver function scores');
    if (data.geneticRisk === 'low') positive.push('Low genetic predisposition');

    // Analyze concerning factors
    if (data.age > 50) concerning.push('Age above 50 increases risk');
    if (data.bmi > 25) concerning.push('BMI above healthy range');
    if (data.alcoholConsumption > 7) concerning.push('High alcohol consumption');
    if (data.smoking === 'yes') concerning.push('Smoking habit');
    if (data.physicalActivity < 3) concerning.push('Low physical activity levels');
    if (data.diabetes === 'yes') concerning.push('Diabetes present');
    if (data.hypertension === 'yes') concerning.push('Hypertension present');
    if (data.liverFunctionScore < 50) concerning.push('Liver function scores below normal');
    if (data.geneticRisk === 'high') concerning.push('High genetic predisposition');

    return { positive, concerning };
  }

  private generateRecommendations(data: InsertPrediction, riskLevel: string): string[] {
    const recommendations: string[] = [];

    // General recommendations
    recommendations.push('Schedule regular liver function monitoring');

    // Risk level specific recommendations
    if (riskLevel === 'low') {
      recommendations.push('Continue current healthy lifestyle');
      recommendations.push('Maintain regular physical activity routine');
      if (data.alcoholConsumption <= 3) {
        recommendations.push('Keep alcohol consumption at current low levels');
      }
    } else if (riskLevel === 'medium') {
      recommendations.push('Consider lifestyle modifications to reduce risk');
      if (data.alcoholConsumption > 3) {
        recommendations.push('Reduce alcohol consumption');
      }
      if (data.physicalActivity < 5) {
        recommendations.push('Increase physical activity to at least 5 hours per week');
      }
      recommendations.push('Schedule more frequent medical check-ups');
    } else {
      recommendations.push('Seek immediate medical consultation');
      recommendations.push('Consider comprehensive liver health evaluation');
      if (data.alcoholConsumption > 7) {
        recommendations.push('Significantly reduce or eliminate alcohol consumption');
      }
      if (data.smoking === 'yes') {
        recommendations.push('Consider smoking cessation programs');
      }
    }

    // Specific condition recommendations
    if (data.diabetes === 'yes') {
      recommendations.push('Maintain optimal blood sugar control');
    }
    if (data.hypertension === 'yes') {
      recommendations.push('Monitor and control blood pressure regularly');
    }
    if (data.bmi > 25) {
      recommendations.push('Work towards achieving healthy weight range');
    }

    return recommendations.slice(0, 5); // Limit to 5 recommendations
  }
}

export const storage = new MemStorage();
