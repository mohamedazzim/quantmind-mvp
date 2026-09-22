"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import { api, ApiError } from "./api";

export interface User {
  user_id: string;
  username: string;
  email: string;
  role: "ADMIN" | "RESEARCHER" | "VIEWER";
  is_active: boolean;
  force_password_change?: boolean;
  created_at: string;
}

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();
  const pathname = usePathname();

  const refreshUser = async () => {
    const token = localStorage.getItem("quantmind_token");
    if (!token) {
      setUser(null);
      setIsLoading(false);
      return;
    }

    try {
      const profile = await api.get<User>("/auth/me");
      setUser(profile);
    } catch (err) {
      localStorage.removeItem("quantmind_token");
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    refreshUser();
  }, []);

  useEffect(() => {
    if (!isLoading && !user && pathname !== "/login") {
      router.push("/login");
    }
  }, [user, isLoading, pathname, router]);

  const login = async (username: string, password: string) => {
    const resp = await api.post<{ access_token: string; user: User }>("/auth/login", {
      username,
      password,
    });
    localStorage.setItem("quantmind_token", resp.access_token);
    setUser(resp.user);
    router.push("/dashboard");
  };

  const logout = () => {
    api.post("/auth/logout").catch(() => {});
    localStorage.removeItem("quantmind_token");
    setUser(null);
    router.push("/login");
  };

  return (
    <AuthContext.Provider value={{ user, isLoading, login, logout, refreshUser }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
