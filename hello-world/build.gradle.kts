

tasks.register("train") {
    doLast {
        exec {
            commandLine("python", "main.py")
        }
    }
}