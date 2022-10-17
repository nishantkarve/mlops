<h3>Overview</h3>

Amazon SageMaker Pipelines , a new capability of Amazon SageMaker  that makes it easy for data scientists and engineers to build, automate, and scale end to end machine learning pipelines. SageMaker Pipelines is a native workflow orchestration tool  for building ML pipelines that take advantage of direct Amazon SageMaker  integration. Three components improve the operational resilience and reproducibility of your ML workflows: pipelines, model registry, and projects. These workflow automation components enable you to easily scale your ability to build, train, test, and deploy hundreds of models in production, iterate faster, reduce errors due to manual orchestration, and build repeatable mechanisms.

SageMaker projects introduce MLOps templates that automatically provision the underlying resources needed to enable CI/CD capabilities for your ML development lifecycle. You can use a number of built-in templates  or create your own custom template (https://docs.aws.amazon.com/sagemaker/latest/dgsagemaker-projects-templates-custom.html ). You can use SageMaker Pipelines independently to create automated workflows; however, when used in combination with SageMaker projects, the additional CI/CD capabilities are provided automatically. The following screenshot shows how the three components of SageMaker Pipelines can work together in an example SageMaker project.


![My Image](mlflow.png)


This lab focuses on using an MLOps template to bootstrap your ML project and establish a CI/CD pattern from sample code. We show how to use the built-in build, train, and deploy project template as a base for a customer churn classification example. This base template enables CI/CD for training ML models, registering model artifacts to the model registry, and automating model deployment with manual approval and automated testing.

<h3>MLOps Template for building, training and deploying modelsHeader anchor link</h3>

We start by taking a detailed look at what AWS services are launched when this build, train, and deploy MLOps template is launched. Later, we discuss how to modify the skeleton for a custom use case.

In SageMaker Studio, you can now choose the Projects menu on the Components and registries menu.


![My Image](images/image1.png)

Once you choose Projects, click on <b>Create project</b> as below:

![My Image](images/image2.png)

On the projects page, you can launch a preconfigured SageMaker MLOps template. For this lab, we choose <b>MLOps template for model building, training, and deployment</b> and click on <b>Select project template</b>

![My Image](images/image3.png)

In the next page provide Project Name and short Description and select <b>Create Project.</b>

![My Image](images/image4.png)

The project will take a while to be created.

![My Image](images/image5.png)

Launching this template starts a model building pipeline by default, and while there is no cost for using SageMaker Pipelines itself, you will be charged for the services launched. Cost varies by Region. A single run of the model build pipeline in us-east-1 is estimated to cost less than $0.50. Models approved for deployment incur the cost of the SageMaker endpoints (test and production) for the Region using an ml.m5.large instance.

After the project is created from the MLOps template, the following architecture is deployed.

![My Image](images/image6.png)

Included in the architecture are the following AWS services and resources:

The MLOps templates that are made available through SageMaker projects are provided via an AWS Service Catalog  portfolio that automatically gets imported when a user enables projects on the Studio domain.

Two repositories are added to AWS CodeCommit :

The first repository provides scaffolding code to create a multi-step model building pipeline including the following steps: data processing, model training, model evaluation, and conditional model registration based on accuracy. As you can see in the pipeline.py file, this pipeline trains a linear regression model using the XGBoost algorithm on the well-known UCI Abalone dataset . This repository also includes a build specification file , used by AWS CodePipeline  and AWS CodeBuild  to run the pipeline automatically.

The second repository contains code and configuration files for model deployment, as well as test scripts required to pass the quality gate. This repo also uses CodePipeline and CodeBuild, which run an AWS CloudFormation  template to create model endpoints for staging and production.

Two CodePipeline pipelines:

The ModelBuild pipeline automatically triggers and runs the pipeline from end to end whenever a new commit is made to the ModelBuild CodeCommit repository.

The ModelDeploy pipeline automatically triggers whenever a new model version is added to the model registry and the status is marked as Approved. Models that are registered with Pending or Rejected statuses aren’t deployed.

An Amazon Simple Storage Service  (Amazon S3) bucket is created for output model artifacts generated from the pipeline.

SageMaker Pipelines uses the following resources: 

This workflow contains the directed acyclic graph (DAG) that trains and evaluates our model. Each step in the pipeline keeps track of the lineage and intermediate steps can be cached for quickly re-running the pipeline. Outside of templates, you can also create pipelines using the SDK .

Within SageMaker Pipelines, the SageMaker model registry  tracks the model versions and respective artifacts, including the lineage and metadata for how they were created. Different model versions are grouped together under a model group, and new models registered to the registry are automatically versioned. The model registry also provides an approval workflow for model versions and supports deployment of models in different accounts. You can also use the model registry through the boto3 package .

Two SageMaker endpoints:

After a model is approved in the registry, the artifact is automatically deployed to a staging endpoint followed by a manual approval step.

If approved, it’s deployed to a production endpoint in the same AWS account.

All SageMaker resources, such as training jobs, pipelines, models, and endpoints, as well as AWS resources listed in this lab, are automatically tagged with the project name and a unique project ID tag.

<h3>Modifying the Seed Code for Custom Use Case</h3>

After your project has been created, the architecture described earlier is deployed and the visualization of the pipeline is available on the Pipelines drop-down menu within SageMaker Studio.

![My Image](images/image7.png)

To modify the sample code from this launched template, <b>we first need to clone the CodeCommit repositories to our local SageMaker Studio instance</b>. From the list of projects, choose the one that was just created. On the <b>Repositories</b> tab, you can select the hyperlinks to locally clone the CodeCommit repos.

![My Image](images/image8.png)

![My Image](images/image9.png)

![My Image](images/image10.png)

Once both repositories have been cloned you should see the following:

![My Image](images/image11.png)

