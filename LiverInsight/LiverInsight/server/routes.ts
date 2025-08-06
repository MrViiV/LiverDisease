import type { Express } from "express";
import { createServer, type Server } from "http";
import { z } from "zod";
import { insertPredictionSchema } from "@shared/schema";
import { storage } from "./storage";

export async function registerRoutes(app: Express): Promise<Server> {
  // Liver disease prediction endpoint
  app.post("/api/predict", async (req, res) => {
    try {
      // Validate request body
      const validationResult = insertPredictionSchema.safeParse(req.body);
      if (!validationResult.success) {
        return res.status(400).json({
          message: "Invalid input data",
          errors: validationResult.error.errors,
        });
      }

      const data = validationResult.data;

      // Calculate risk score using prediction algorithm
      const result = await storage.predictLiverDisease(data);

      res.json(result);
    } catch (error) {
      console.error("Prediction error:", error);
      res.status(500).json({
        message: "Internal server error during prediction",
      });
    }
  });

  // Get prediction history (optional)
  app.get("/api/predictions", async (req, res) => {
    try {
      const predictions = await storage.getAllPredictions();
      res.json(predictions);
    } catch (error) {
      console.error("Error fetching predictions:", error);
      res.status(500).json({
        message: "Failed to fetch prediction history",
      });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
