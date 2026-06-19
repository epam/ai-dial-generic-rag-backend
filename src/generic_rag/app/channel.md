The channel is accessible via instance of Schema-rich Application created in DIAL.

The `application_id` is the same name as would be normally used in chat completion requests, and can be one of the following:
* for applications within the organization it's `{deployment_name}` (for example: `generic-rag-app`)
* for custom applications created in user's bucket it's `applications/{bucket}/{application_name}` 
  (for example: `applications/N8j2M5hCySoyTFDGRU2F3U/generic-rag-app`)
