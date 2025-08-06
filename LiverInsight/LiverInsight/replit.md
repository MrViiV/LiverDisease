# Overview

This is a liver disease prediction application that uses AI-powered medical assessment to evaluate liver health risk factors. The application provides a comprehensive health questionnaire interface where users can input medical data and receive risk assessments with personalized recommendations. It's designed as a clinical screening tool to assist healthcare professionals in preliminary liver disease risk evaluation.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: React with TypeScript, built using Vite for fast development and builds
- **UI Library**: Shadcn/ui components built on Radix UI primitives for accessibility and consistency
- **Styling**: Tailwind CSS with custom medical theme colors and CSS variables for design system consistency
- **State Management**: React Hook Form for form handling with Zod validation, TanStack Query for server state management
- **Routing**: Wouter for lightweight client-side routing

## Backend Architecture
- **Framework**: Express.js with TypeScript for RESTful API endpoints
- **Prediction Engine**: Custom liver disease risk calculation algorithm implemented in the storage layer
- **API Design**: RESTful endpoints for prediction submission (/api/predict) and history retrieval (/api/predictions)
- **Data Validation**: Zod schemas for request/response validation with strict medical parameter constraints
- **Error Handling**: Centralized error middleware with proper HTTP status codes and medical context

## Data Storage Solutions
- **Database**: PostgreSQL configured via Drizzle ORM for type-safe database operations
- **Schema Management**: Drizzle Kit for database migrations and schema evolution
- **Connection**: Neon Database serverless PostgreSQL for cloud-native scalability
- **Fallback Storage**: In-memory storage implementation for development and testing scenarios

## Development Architecture
- **Monorepo Structure**: Shared schema definitions between client and server in /shared directory
- **Build System**: Vite for frontend bundling, ESBuild for server bundling with external package handling
- **Development Server**: Hot reload with Vite middleware integration and development banner for Replit environment
- **Type Safety**: Strict TypeScript configuration with path mapping for clean imports

## Medical Algorithm Design
- **Risk Calculation**: Multi-factor analysis considering age, BMI, lifestyle factors, medical history, and liver function scores
- **Risk Stratification**: Three-tier system (low/medium/high) with corresponding color coding and visual indicators
- **Recommendation Engine**: Personalized health recommendations based on individual risk factors and assessment results
- **Clinical Validation**: Algorithm designed for preliminary screening with appropriate medical disclaimers

# External Dependencies

## Database Services
- **Neon Database**: Serverless PostgreSQL hosting with connection pooling and automatic scaling
- **Drizzle ORM**: Type-safe database operations with PostgreSQL dialect support

## UI and Component Libraries
- **Radix UI**: Comprehensive set of accessible, unstyled UI primitives for form controls, dialogs, and interactive elements
- **Shadcn/ui**: Pre-built component library with consistent design system and Tailwind integration
- **Lucide React**: Medical and general-purpose icons for intuitive user interface elements

## Form and Validation
- **React Hook Form**: Performance-optimized form handling with minimal re-renders
- **Zod**: Runtime type validation for medical data with comprehensive error messaging
- **Hookform Resolvers**: Integration layer between React Hook Form and Zod validation

## Development and Build Tools
- **Vite**: Fast development server and build tool with React plugin support
- **TanStack Query**: Server state management for API calls with caching and background updates
- **Date-fns**: Date manipulation utilities for timestamp handling and formatting
- **Replit Integration**: Development environment plugins for error overlay and cartographer mapping

## Styling and Design
- **Tailwind CSS**: Utility-first CSS framework with custom medical theme configuration
- **Class Variance Authority**: Type-safe variant handling for component styling
- **Embla Carousel**: Touch-friendly carousel component for result presentation