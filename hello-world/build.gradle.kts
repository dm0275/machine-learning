

tasks.register("trainModel") {
    doLast {
        exec {
            commandLine("python", "train.py")
        }
    }
}

tasks.register("loadModel") {
    doLast {
        exec {
            commandLine("python", "load.py")
        }
    }
}