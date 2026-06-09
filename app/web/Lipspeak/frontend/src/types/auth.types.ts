/** auth.types.ts
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Authentication related TypeScript types and interfaces.
 * @date: 09 June 2026
 * @returns: Authentication type definitions.
 *
 */


/* -------------------------------------------------------------------------- */
/*                                  User Types                                */
/* -------------------------------------------------------------------------- */

export interface User {
  id: number;
  username: string;
  name: string;
  created_at?: string;
}


/* -------------------------------------------------------------------------- */
/*                              Request Payloads                              */
/* -------------------------------------------------------------------------- */

export interface LoginRequest {
  username: string;
  password: string;
}

export interface SignupRequest {
  username: string;
  name: string;
  password: string;
}


/* -------------------------------------------------------------------------- */
/*                              Response Payloads                             */
/* -------------------------------------------------------------------------- */

export interface LoginResponse {
  token: string;
  user: User;
}

export interface SignupResponse {
  message: string;
}


/* -------------------------------------------------------------------------- */
/*                                 Auth State                                 */
/* -------------------------------------------------------------------------- */

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
}


/* -------------------------------------------------------------------------- */
/*                                 API Errors                                 */
/* -------------------------------------------------------------------------- */

export interface ApiError {
  message: string;
  statusCode?: number;
}


/* -------------------------------------------------------------------------- */
/*                                 Login Form Data                            */
/* -------------------------------------------------------------------------- */


export interface LoginFormData {
  username: string;
  password: string;
}



/* -------------------------------------------------------------------------- */
/*                                 Signup Form Data                           */
/* -------------------------------------------------------------------------- */


export interface SignupFormData {
  username: string;
  name: string;
  password: string;
  confirmPassword: string;
}