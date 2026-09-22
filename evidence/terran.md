---
title: Terran Enterprise
tags: engineering, typescript, postgres, payments
---

# Full-stack engineering intern, November 2025 to now, New York (remote)

Built and launched Terran Denizen, a career platform for MBA students at the M7 business schools, from scratch as a founding engineer: four Next.js apps (public site, member, admin and partner portals) and a Node API on one shared PostgreSQL backend, with the iOS app now in progress in Expo (React Native).

Replaced hardcoded business rules and data with dashboard configuration: school tiers and eligibility, application questions, hiring-pipeline stages and pricing all change without a deploy.

Pushed authorization into the database itself with 76 row-level security policies, guarded by an hourly job that queries 29 sensitive tables with the public API key and fails the build on any leak.

Reconciled payments across Stripe's webhook and redirect paths so no captured payment goes unrecorded, and centralized every discount rule in one resolver shared by client and server.

Extended the hourly check from anonymous access to signed-in users, and it caught two unrestricted write policies.
