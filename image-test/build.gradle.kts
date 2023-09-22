/**
 * Image ML Tasks
 */

tasks.register("trainModel") {
    doLast {
        exec {
            commandLine("python", "train.py")
        }
    }
}
