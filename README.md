---
title: Movie Recommendation System
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.24.0
app_file: app.py
pinned: false
---

# Advanced Movie Recommendation System

A sophisticated movie recommendation system that combines BERT embeddings, CLIP vision-language model, and BART text generation to provide personalized movie recommendations.

## Features
- Content-based recommendations using plot analysis
- Personality-based recommendations
- AI-powered hybrid recommendations
- Movie poster analysis
- Multi-modal recommendation engine

Built with Streamlit and Hugging Face models.

## Core Features & Models

### 1. Content-Based Analysis
- **BERT Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`)
  - Creates semantic embeddings of movie plots
  - Finds similar movies based on plot descriptions
  - Enables deep thematic matching

- **CLIP Vision-Language Model** (`openai/clip-vit-base-patch32`)
  - Analyzes movie posters and visual elements
  - Matches visual styles across movies
  - Enables poster-based recommendations

### 2. Natural Language Understanding
- **BART Text Generation** (`facebook/bart-large-cnn`)
  - Generates personalized movie explanations
  - Provides context-aware recommendations
  - Creates natural language responses

- **Zero-Shot Classification** (`facebook/bart-large-mnli`)
  - Analyzes movie themes without training
  - Determines plot complexity
  - Classifies genres dynamically

### 3. Multilingual Support
- **mBART Translation** (`facebook/mbart-large-50-many-to-many-mmt`)
  - Handles multiple languages
  - Enables cross-language recommendations
  - Translates movie descriptions

## How It Works

1. **Movie DNA Analysis**
   - Extracts thematic elements
   - Analyzes plot complexity
   - Identifies genre patterns
   - Creates visual fingerprints

2. **Personalization Engine**
   - Mood-based filtering
   - Genre preference matching
   - Complexity alignment
   - Visual style matching

3. **Hybrid Recommendation**
   - Combines content-based and personality-based features
   - Uses LLMs for explanation generation
   - Provides multi-modal analysis

## Technical Implementation

1. **Data Processing**
   - Movie metadata analysis
   - Plot embedding generation
   - Poster image processing
   - Multi-modal feature fusion

2. **Model Pipeline**
   - Sequential processing
   - Multi-modal integration
   - Efficient caching
   - Streamlit-optimized rendering

## Usage

1. **Content-Based Search**
   - Enter favorite movies
   - Get similar recommendations
   - View visual matches

2. **Personality Quiz**
   - Set mood preferences
   - Select genre interests
   - Specify complexity level

3. **AI-Powered Recommendations**
   - Natural language preferences
   - Detailed explanations
   - Multi-factor matching

## Deployment

Hosted on Hugging Face Spaces with Streamlit integration.
