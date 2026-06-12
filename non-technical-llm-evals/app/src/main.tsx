import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { loadConfigOverrides } from './utils/config.ts'

// Apply config.json overrides to DEFAULT_SETTINGS before rendering so new
// conversations are created with the deployed defaults.
loadConfigOverrides().finally(() => {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
})
