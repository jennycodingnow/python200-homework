# Part 3: Project

## Part A: Supabase Setup

### Step 1: Create Your Account and Project

completed

### Step 2: Locate Your Credentials

completed

### Step 3: Create a .env File

completed

### Step 4: Create the Tables

completed

### Step 5: Confirm

### Write a sentence confirming your project is set up (or note any issues you ran into).

Answer:  
I think the project is set up, but I was a little confused about whether I configured Supabase correctly during the initial setup. When I created the project in Supabase, I wasn’t sure whether I needed to connect it to GitHub at that point.

## Part B: Cloud Cost Analysis

### Write a short summary (a few sentences to a paragraph) covering:

What each scenario costs, and whether the numbers surprised you:

- Scenario A — Lightweight compute: A t3.micro EC2 instance (1 vCPU, 1 GB RAM), on-demand pricing, running 8 hours per day, 5 days per week (approximately 160 hours per month). Use the US East (N. Virginia) region.

- Scenario B — Heavy analytics workload: A p3.2xlarge EC2 instance (8 vCPU, 1 V100 GPU), running 24/7 for the full month (730 hours); an RDS db.m5.large instance (2 vCPU, 8 GB RAM); and an S3 Standard storage bucket with 1 TB of data. Use US East (N. Virginia).

Answer:  
Scenario A — Lightweight compute: The t3.micro EC2 instance costs $1.66 per month, or approximately $19.92 for 12 months. For a simple configuration like this, the cost did not surprise me because the instance has limited resources and only runs for about 160 hours per month. It is relatively inexpensive for a basic workload.

Scenario B — Heavy analytics workload: The p3.2xlarge EC2 instance costs $2,233.80 per month, or $26,805.60 for 12 months. The RDS db.m5.large instance costs 181.59 USD per month, or 2,179.08 for 12 months. The S3 Standard storage costs $23.55 per month, or $282.60 for 12 months. The total for Scenario B is approximately $2,438.94 per month, or $29,267.28 for 12 months. I found it interesting how quickly AWS costs can increase when using powerful resources such as GPU instances. It is important to use billing alarms, automated cost controls, and preventative rules to avoid unexpected charges.

Comparison: Scenario A is much cheaper than Scenario B. The biggest difference comes from the p3.2xlarge instance and its V100 GPU. This shows that GPU instances can be very expensive, but they can be worth the cost when they are needed for demanding workloads such as machine learning, AI model training, or other tasks that benefit from GPU processing. For simple applications that do not require GPU processing, a smaller CPU based instance would be a much more cost effective choice.

## The Video

completed
