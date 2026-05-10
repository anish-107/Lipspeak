// src/hooks/useAuth.ts

/**
 * @authors: Anish Kumar, Bidipta Barua, Dibyasmita Hati, Arpan Haldar
 * @description: Authentication hook handling login and signup form logic.
 * @date: 10 May 2026
 * @returns: Authentication handlers and states.
 */

"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

interface LoginFormState {
  email: string;
  password: string;
}

interface SignupFormState {
  name: string;
  email: string;
  password: string;
}

export function useAuth() {
  const router = useRouter();

  const [loginLoading, setLoginLoading] = useState(false);
  const [signupLoading, setSignupLoading] = useState(false);

  const [loginForm, setLoginForm] = useState<LoginFormState>({
    email: "",
    password: "",
  });

  const [signupForm, setSignupForm] = useState<SignupFormState>({
    name: "",
    email: "",
    password: "",
  });

  const handleLoginChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setLoginForm((previous) => ({
      ...previous,
      [event.target.name]: event.target.value,
    }));
  };

  const handleSignupChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setSignupForm((previous) => ({
      ...previous,
      [event.target.name]: event.target.value,
    }));
  };

  const handleLoginSubmit = async (
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault();

    try {
      setLoginLoading(true);

      /**
       * Temporary development authentication.
       * Replace with real backend API later.
       */

      document.cookie = "token=demo-token; path=/";

      router.push("/dashboard");
    } catch (error) {
      console.error(error);
    } finally {
      setLoginLoading(false);
    }
  };

  const handleSignupSubmit = async (
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault();

    try {
      setSignupLoading(true);

      /**
       * Temporary development authentication.
       * Replace with real backend API later.
       */

      document.cookie = "token=demo-token; path=/";

      router.push("/dashboard");
    } catch (error) {
      console.error(error);
    } finally {
      setSignupLoading(false);
    }
  };

  return {
    loginForm,
    signupForm,
    loginLoading,
    signupLoading,
    handleLoginChange,
    handleSignupChange,
    handleLoginSubmit,
    handleSignupSubmit,
  };
}