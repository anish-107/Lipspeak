/** auth.service.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Authentication API service functions.
 * @date: 09 June 2026
 * @returns: Authentication related API methods.
 *
 */


// Imports
import { apiClient } from "@/lib/api-client";
import type {
  LoginRequest,
  LoginResponse,
  SignupRequest,
  SignupResponse,
  User,
} from "@/types/auth.types";


// Auth Service
export const authService = {
  /* ------------------------------------------------------------------------ */
  /*                                   Login                                  */
  /* ------------------------------------------------------------------------ */

  async login(data: LoginRequest): Promise<LoginResponse> {
    const response = await apiClient.post<LoginResponse>(
      "/auth/login",
      data,
    );

    return response.data;
  },


  /* ------------------------------------------------------------------------ */
  /*                                 Register                                 */
  /* ------------------------------------------------------------------------ */

  async signup(data: SignupRequest): Promise<SignupResponse> {
    const response = await apiClient.post<SignupResponse>(
      "/auth/register",
      data,
    );

    return response.data;
  },


  /* ------------------------------------------------------------------------ */
  /*                               Current User                               */
  /* ------------------------------------------------------------------------ */

  async getCurrentUser(): Promise<User> {
    const response = await apiClient.get<User>(
      "/auth/me",
    );

    return response.data;
  },


  /* ------------------------------------------------------------------------ */
  /*                                  Logout                                  */
  /* ------------------------------------------------------------------------ */

  logout(): void {
    localStorage.removeItem("access_token");
  },
};