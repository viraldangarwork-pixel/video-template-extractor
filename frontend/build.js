const fs = require('fs');
const path = require('path');

// 1. Read from system / Vercel environment variables
let backendUrl = process.env.VTPL_BACKEND_URL || process.env.NEXT_PUBLIC_BACKEND_URL || process.env.BACKEND_URL || '';

// 2. Read from .env if present in the frontend directory
const envPath = path.join(__dirname, '.env');
if (!backendUrl && fs.existsSync(envPath)) {
  const envContent = fs.readFileSync(envPath, 'utf-8');
  for (const line of envContent.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const match = trimmed.match(/^(?:VTPL_BACKEND_URL|NEXT_PUBLIC_BACKEND_URL|BACKEND_URL)\s*=\s*['"]?(.*?)['"]?$/);
    if (match && match[1]) {
      backendUrl = match[1];
      break;
    }
  }
}

// Normalize backend URL: strip whitespace and trailing slashes
backendUrl = (backendUrl || '').trim().replace(/\/+$/, '');

const output = `// VTPL Frontend Configuration
// Auto-generated during build from .env / Vercel Environment Variables (VTPL_BACKEND_URL)
window.VTPL_BACKEND_URL = "${backendUrl}";
`;

const configJsPath = path.join(__dirname, 'config.js');
fs.writeFileSync(configJsPath, output, 'utf-8');

console.log(`[VTPL Build] Injected backend URL: ${backendUrl ? `"${backendUrl}"` : '"" (defaults to relative Vercel rewrites)'}`);
