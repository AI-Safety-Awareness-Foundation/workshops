# AI Safety Workshop: Agentic Fraud Demo

This is a simple frontend application that demonstrates concepts related to AI safety, specifically focusing on agentic fraud scenarios. The application provides a chat interface for interacting with Claude 3.7, Anthropic's advanced AI assistant.

## Purpose

This project is created for **educational and research purposes only**. It aims to help participants understand:

- How AI systems might be exploited for fraudulent purposes
- Techniques for recognizing and preventing AI-based fraud
- The importance of proper safeguards in AI deployments

## Getting Started

### Prerequisites

- Node.js (v14 or later)
- npm or yarn

### Installation

1. Clone the repository
2. Install dependencies:

```bash
npm install
```

3. Set up your Anthropic API key:
   - Create a `.env` file in the root directory by copying `.env.example`
   - Add your Anthropic API key to the `.env` file:
   ```
   REACT_APP_ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

> **Important**: Never commit your API key to version control.

### Running the Application

Start the development server:

```bash
npm start
```

This will run the app in development mode. Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

### Building for Production

To build the app for production:

```bash
npm run build
```

This creates optimized files in the `build` folder ready for deployment.

## Project Structure

- `public/` - Static assets and HTML template
- `src/` - React application source code
  - `components/` - Reusable UI components
  - `App.js` - Main application component
  - `index.js` - Application entry point

## Educational Notes

This application is designed to illustrate concepts from the AI safety workshop. Remember:

1. All features are designed for educational purposes only
2. The examples shown should not be replicated in production systems
3. The goal is to increase awareness of potential risks and mitigation strategies

## License

This project is part of an educational workshop and should be used accordingly.