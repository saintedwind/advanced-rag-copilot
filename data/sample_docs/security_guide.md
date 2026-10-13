# Security and Data Handling Guide

## Encryption
Customer data is encrypted in transit using TLS 1.2 or newer and at rest using AES-256. Encryption keys are managed through a dedicated key-management service. Enterprise customers can rotate their own keys every 90 days; the default managed rotation period is 180 days.

## Identity and access
The platform supports SAML 2.0 single sign-on, SCIM user provisioning, and role-based access control. Administrative roles should require multi-factor authentication. Audit events are retained for 365 days on Business and Mission Critical plans, and for 90 days on Standard.

## Incident communication
For a confirmed security incident affecting customer data, Northstar notifies the designated security contact without undue delay and provides regular updates while investigation is ongoing. Customers can export audit logs through the administration console in JSON Lines format.
