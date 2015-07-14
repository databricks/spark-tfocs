organization := "com.databricks"

name := "spark-tfocs"

version := "1.0-SNAPSHOT"

spName := "databricks/spark-tfocs"

sparkVersion := "1.4.0"

sparkComponents += "mllib"

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

credentials += Credentials(Path.userHome / ".ivy2" / ".credentials")

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "2.1.5" % Test
)

parallelExecution in Test := false

scalariformSettings
