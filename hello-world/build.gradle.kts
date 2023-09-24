/**
 * Hello World ML Tasks
 */

tasks.register("trainModel") {
    doLast {
        exec {
            commandLine("python", "train.py")
        }
    }
}

tasks.register("testModel") {
    doLast {
        exec {
            commandLine("python", "predict.py")
        }
    }
}

tasks.register("trainModelNoDeps") {
    doLast {
        exec {
            commandLine("python", "train_nodeps.py")
        }
    }
}