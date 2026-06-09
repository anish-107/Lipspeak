/** api-client.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Centralized Axios client for API communication.
 * @date: 09 June 2026
 * @returns: Configured Axios instance.
 *
 */


// Imports
import axios from "axios";
import { env } from "@/config/env";


// API Client
export const apiClient = axios.create({
  baseURL: env.API_URL,

  timeout: 30000,

  headers: {
    "Content-Type": "application/json",
  },
});


/* -------------------------------------------------------------------------- */
/*                          Request Interceptor                               */
/* -------------------------------------------------------------------------- */

apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("access_token");

    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
  },

  (error) => Promise.reject(error),
);


/* -------------------------------------------------------------------------- */
/*                          Response Interceptor                              */
/* -------------------------------------------------------------------------- */

apiClient.interceptors.response.use(
  (response) => response,

  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("access_token");
    }

    return Promise.reject(error);
  },
);