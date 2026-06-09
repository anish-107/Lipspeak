/** env.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Centralized environment variable configuration.
 * @date: 09 June 2026
 * @returns: Environment configuration object.
 *
 */


/* -------------------------------------------------------------------------- */
/*                           Environment Variables                            */
/* -------------------------------------------------------------------------- */

export const env = {
  APP_NAME: process.env.NEXT_PUBLIC_APP_NAME ?? "LipSpeak AI",

  API_URL:
    process.env.NEXT_PUBLIC_API_URL ??
    "http://localhost:8000/api",

  WS_URL:
    process.env.NEXT_PUBLIC_WS_URL ??
    "ws://localhost:8000/ws",
};