/*
Configuration Terraform
*/

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "Région AWS"
  type        = string
  default     = "eu-west-3"
}

variable "instance_type" {
  description = "Type d'instance EC2"
  type        = string
  default     = "g4dn.xlarge"
}

resource "aws_instance" "whisper_gpu" {
  ami                    = "ami-0809e1e48f650e1f9"  # Ubuntu 22.04 LTS eu-west-3 version du 22-08-25
  instance_type          = "g4dn.xlarge"
  key_name              = "summora-keypair-190625"
  vpc_security_group_ids = [aws_security_group.whisper_sg.id]
  availability_zone      = "eu-west-3c"
  user_data = base64encode(templatefile("${path.module}/setup-whisper.sh", {}))

  # Désactive la protection contre l'arrêt (pour pouvoir l'arrêter automatiquement)
  disable_api_termination = false

  tags = {
    Name = "summora-whisper-gpu"
    Project = "summora"
  }
}

resource "aws_security_group" "whisper_sg" {
  name_prefix = "summora-whisper-"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Outputs
output "instance_id" {
  value = aws_instance.whisper_gpu.id
}

output "public_ip" {
  value = aws_instance.whisper_gpu.public_ip
}
