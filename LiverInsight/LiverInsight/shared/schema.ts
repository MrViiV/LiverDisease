import { sql } from "drizzle-orm";
import { pgTable, text, varchar, real, integer } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const predictions = pgTable("predictions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  age: integer("age").notNull(),
  gender: text("gender").notNull(),
  bmi: real("bmi").notNull(),
  alcoholConsumption: real("alcohol_consumption").notNull(),
  smoking: text("smoking").notNull(),
  geneticRisk: text("genetic_risk").notNull(),
  physicalActivity: real("physical_activity").notNull(),
  diabetes: text("diabetes").notNull(),
  hypertension: text("hypertension").notNull(),
  liverFunctionScore: real("liver_function_score").notNull(),
  riskScore: real("risk_score").notNull(),
  riskLevel: text("risk_level").notNull(),
  createdAt: text("created_at").default(sql`CURRENT_TIMESTAMP`),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export const insertPredictionSchema = createInsertSchema(predictions).omit({
  id: true,
  riskScore: true,
  riskLevel: true,
  createdAt: true,
}).extend({
  age: z.number().min(1).max(100),
  gender: z.enum(["male", "female"]),
  bmi: z.number().min(10).max(50),
  alcoholConsumption: z.number().min(0).max(20),
  smoking: z.enum(["yes", "no"]),
  geneticRisk: z.enum(["low", "medium", "high"]),
  physicalActivity: z.number().min(0).max(20),
  diabetes: z.enum(["yes", "no"]),
  hypertension: z.enum(["yes", "no"]),
  liverFunctionScore: z.number().min(0).max(100),
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type InsertPrediction = z.infer<typeof insertPredictionSchema>;
export type Prediction = typeof predictions.$inferSelect;
