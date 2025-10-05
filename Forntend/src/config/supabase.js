import { createClient } from "@supabase/supabase-js";

const supabaseUrl = "https://nftwaneaeexlzslbipqc.supabase.co";
const supabaseKey = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5mdHdhbmVhZWV4bHpzbGJpcHFjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk2NTE1OTMsImV4cCI6MjA3NTIyNzU5M30.OMzT3KfcvWTZA755rPndADjfHdEjKbO92Ouonbyr4VY";

export const supabase = createClient(supabaseUrl, supabaseKey);
