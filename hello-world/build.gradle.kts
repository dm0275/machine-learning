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

tasks.register("runModel") {
    doLast {
        exec {
            commandLine("python", "predict.py")
        }
    }
}