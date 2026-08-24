# Part 1: Warmup — Cloud Concepts

## Cloud Concepts Question 1

### What is the core economic model of cloud computing, and how does it differ from owning your own servers?

The core The core economic model of cloud computing is a pay-as-you-use model, where you pay according to the resources you use. This differs from owning your own servers because owning servers requires purchasing and maintaining physical hardware. With cloud computing, the physical infrastructure is owned and managed by the cloud provider, and you can spin up resources when needed and pay for the resources you use. This can save money compared with owning your own servers because you do not have to purchase and maintain the physical hardware yourself.

## Cloud Concepts Question 2

### What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

Vertical scaling is upgrading one machine to make it more powerful by adding more RAM, CPU, or GPU. For example, you might choose vertical scaling when you need a more powerful machine to handle a demanding workload.

Horizontal scaling is adding more machines and distributing the work across them. For example, you might choose horizontal scaling when traffic to a website increases and you need additional servers to handle the traffic.

#### Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.

- A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.

This would be a horizontal scaling because sudden spike in traffic needs adding more machines to distribute the traffic.

- A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.

This will a vertical scaling because it needs a faster GPU and more RAM to a machine.

- A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.

This will be horizontal scaling because it is a spike in workload and the work can be split across machines.

## Cloud Concepts Question 3

### Before writing your definitions, classify each item in the list below as IaaS, PaaS, SaaS, or BaaS. One sentence of reasoning is enough for each.

- Gmail: SaaS — Gmail is a complete application that is built, managed, and maintained by Google, so users can simply use it without managing the underlying infrastructure.

- Azure Virtual Machines: IaaS — Azure provides the virtual machine infrastructure, while users are responsible for configuring the operating system, installing software, and managing updates and security.

- AWS S3: IaaS — S3 provides cloud storage infrastructure where users can store and retrieve data without having to manage the physical storage hardware themselves.

- GitHub Codespaces: PaaS — GitHub provides a managed development environment where users can write and run code without having to set up and maintain the underlying development infrastructure.

- Snowflake: PaaS — Snowflake provides a managed cloud data platform where users can store, process, and analyze data without managing the underlying servers and infrastructure.

- Supabase: BaaS — Supabase provides managed backend services such as databases, authentication, and APIs, allowing developers to build applications without creating and managing the backend infrastructure themselves.

### Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.

IaaS – Infrastructure as a Service
IaaS gives users basic computing resources, such as a virtual machine, storage, and a network. The user is responsible for choosing the operating system, installing software, setting up the environment, and managing security updates. For example, AWS EC2.

PaaS - Platform as a Service
PaaS provides a platform where developers can build and run their applications. The provider manages the servers, operating system, and other infrastructure. The developer is mainly responsible for their own code and application. For example, Google App Engine.

SaaS – Software as a Service
SaaS is software that is already built and managed by a provider. Users simply log in and use the software without managing servers or installing the application. For example, Gmail. As a developer/user, there is very little infrastructure to manage.

## Cloud Concepts Question 4

### What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like AWS or GCP directly? What do you gain, and what do you give up?

A managed data platform like Databricks or Snowflake runs on top of cloud providers like AWS, GCP, or Azure. They are not separate cloud providers. Instead, they provide a managed layer that handles many of the cloud resources for you. This makes it faster and easier to get started with large-scale data processing, analytics, and machine learning.
With a managed data platform, you gain simplicity, faster setup, and less work managing infrastructure. However, you give up some flexibility and control, and it can potentially cost more.

Cloud providers like AWS and GCP give users the full set of tools, such as compute, storage, networking, databases, and machine learning services. This gives users more control and flexibility, but they have to build and configure more of the system themselves. This takes more time and technical knowledge.

## Cloud Concepts Question 5

### The lesson names two situations where the cloud is probably not the right choice. What are they?

Two situations where the cloud is not the right choice are when users have data that can be comfortably stored on a single machine and do not need massive computing power. In this case, local processing can be cheaper and faster.

The second situation is when users are starting a project or building an initial prototype. Using the cloud can take more time because they have to learn how to set up and manage the resources.

# Part 2: Warmup — Cloud Landscape

## Cloud Landscape Question 1

### Name the three hyperscalers. For each, write one sentence describing its primary strength and the type of organization most likely to use it.

Amazon Web Services (AWS): AWS is the largest and most widely used cloud provider, with a very large range of services. It is commonly used by startups, large companies, and organizations with engineering teams.

Google Cloud Platform (GCP): GCP is especially strong in data analytics, machine learning, and large-scale computing. It is often used by organizations that work with large amounts of data or AI/ML.

Microsoft Azure: Azure is especially strong for businesses and governments that already use Microsoft products. It is commonly used by large companies, nonprofits, and public-sector organizations that rely on Microsoft 365, Windows, and Active Directory.

## Cloud Landscape Question 2

### The lesson explains why this course switched from Microsoft Azure to Supabase. It gives three concrete reasons. Summarize each reason in your own words — one sentence each.

Access- Supabase is easier for students to access because they can create their own accounts in under two minutes, and its free tier is enough for the course.

Pedagogical fit- Azure Blob Storage stores data as opaque files organized by path, while Supabase stores data as rows and columns in a relational database, which teaches skills that are more transferable.

Pipeline coherence- Supabase uses two related tables, a raw zone and an enriched zone, which makes the pipeline stages easier to understand, inspect, and debug.

### Then add your own reflection: what does this suggest about how you should evaluate a cloud tool when starting a new project?

This suggests that I should look for a cloud tool that is both simple and functional when starting a new project. I should consider whether the tool requires organizational provisioning or allows self-provisioning, especially when there are time constraints. I should also consider whether the skills I learn can transfer to other cloud tools and whether the tool is easy to use, inspect, and debug.

## Cloud Landscape Question 3

### For each of the four scenarios below, identify which service category from the taxonomy table applies (e.g., "object storage", "managed relational DB", "LLM API", "serverless compute") and name one specific provider or product that offers it.

- You need to store 10 TB of image files and retrieve them by filename from any machine.
  Object storage - GCP Cloud Storage

- You need to run an ML training job on a GPU for four hours, then shut it down.
  ML platform - Azure ML

- You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets.
  Compute - AWS Lambda

- You need to send structured data to a large language model and get a text response back.
  LLM API- Azure OpenAI

## Cloud Landscape Question 4

### The lesson says most projects don't use one provider for everything. Describe a simple data project of your own design (one or two sentences is fine) and sketch a plausible stack using services from at least two different providers or products from the taxonomy table. Then answer: is there a benefit to consolidating to one provider, and what would you give up if you did?

A simple data project could be a chatbot that uses an LLM API, such as Azure OpenAI, to answer questions about application data stored in a managed relational database like GCP Cloud SQL. If the project were instead an analyst dashboard using large amounts of historical data, I could use a data warehouse like AWS Redshift. Consolidating everything with one provider can make the project easier to manage, but I would give up flexibility, potentially lower costs from other providers, and independence from one provider.
