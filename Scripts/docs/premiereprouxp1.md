### Write and Read Files to Desktop using Node.js fs (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

This JavaScript code snippet shows how to export data to the user's Desktop and read data from a file on the Desktop using the Node.js 'fs' module. It includes error handling and OS-specific path examples. Requires 'localFileSystem' with 'fullAccess' in the manifest.

```javascript
const fs = require("fs");

async function exportToDesktop() {
	// Export data to the user's Desktop
	try {
		const exportData = "Exported data from plugin";

		// For macOS
		await fs.writeFile(
			"/Users/user/Desktop/export.txt", // 👇 update with your user
			exportData,
			{ encoding: "utf-8" },
		);
		// For Windows, use: "C:/Users/user/Desktop/export.txt"

		console.log("File exported to Desktop");
	} catch (e) {
		console.error("Failed to export file:", e);
	}
}

async function readFromDesktop() {
	// Read a file from the user's Desktop folder
	try {
		// For macOS
		const content = await fs.readFile(
			"/Users/user/Desktop/export.txt", // 👇 update with your user
			"utf8",
		);
		// For Windows, use: "C:/Users/user/Desktop/export.txt"

		console.log("File content:", content);
	} catch (e) {
		console.error("Failed to read file:", e);
	}
}
```

---

### Install, Remove, List UXP Plugins via UPIA on Windows

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/distribution/install

This snippet shows how to manage UXP plugins on Windows using the UnifiedPluginInstallerAgent.exe command-line tool. It includes commands for installing, removing, and listing plugins, requiring administrator privileges and correct file paths.

```batch
cd "C:\Program Files\Common Files\Adobe\Adobe Desktop Common\RemoteComponents\UPI\UnifiedPluginInstallerAgent"

UnifiedPluginInstallerAgent.exe /help
UnifiedPluginInstallerAgent.exe /version
UnifiedPluginInstallerAgent.exe /install <extension-file-path>
UnifiedPluginInstallerAgent.exe /remove <extension-file-path>
UnifiedPluginInstallerAgent.exe /list <all || product display name>

# Examples:
UnifiedPluginInstallerAgent.exe /install "C:\Temp\Test-xjluvc_premierepro.ccx"
UnifiedPluginInstallerAgent.exe /remove "startup-test"
UnifiedPluginInstallerAgent.exe /list all
```

---

### Installing and Importing Spectrum Web Component Button

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum

Shows the necessary steps to use Adobe Spectrum Web Components (SWC) in UXP. This involves installing the component via npm and then importing it before use.

```bash
npm i @swc-uxp-wrappers/button
```

```javascript
import "@swc-uxp-wrappers/button/sp-button.js";
```

---

### Install, Remove, List UXP Plugins via UPIA on macOS

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/distribution/install

This snippet demonstrates how to use the UnifiedPluginInstallerAgent on macOS to manage UXP plugins. It covers installing, removing, and listing plugins using command-line arguments. Ensure you have the correct paths to the extension files and necessary admin privileges.

```shell
cd "/Library/Application Support/Adobe/Adobe Desktop Common/RemoteComponents/UPI/UnifiedPluginInstallerAgent/UnifiedPluginInstallerAgent.app/Contents/macOS"

./UnifiedPluginInstallerAgent --help
./UnifiedPluginInstallerAgent --version
./UnifiedPluginInstallerAgent --install <extension-file-path>
./UnifiedPluginInstallerAgent --remove <extension-file-path>
./UnifiedPluginInstallerAgent --list <all || product display name>

# Examples:
./UnifiedPluginInstallerAgent --install "~/Desktop/Test-xjluvc_premierepro.ccx"
./UnifiedPluginInstallerAgent --remove "startup-test"
./UnifiedPluginInstallerAgent --list all
```

---

### Working with Premiere Application Objects

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference

This section details how to access and interact with core Premiere Pro objects like the application, project, and sequence using the UXP API. It provides examples for getting the active project and active sequence.

````APIDOC
## Working with Premiere Objects

### Premiere Application

The `app` object provides access to the rest of Premiere's objects and methods.

#### Getting the Active Project

To get the currently active project:

```javascript
const project = await app.Project.getActiveProject();
````

#### Getting the Active Sequence

From the project object, you can get the active sequence:

```javascript
const sequence = await project.getActiveSequence();
```

````

--------------------------------

### Sliding Element Example (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLSelectElement

Example demonstrating how to implement a sliding element using pointer events and pointer capture. It includes functions for starting, stopping, and handling pointer movements to update the element's position.

```javascript
// HTML
<style>
    div {
        width: 140px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #fbe;
        position: absolute;
    }
</style>
<div id="slider">SLIDE ME</div>


// JS
function beginSliding(e) {
     slider.setPointerCapture(e.pointerId);
     slider.addEventListener("pointermove", slide);
 }


 function stopSliding(e) {
     slider.releasePointerCapture(e.pointerId);
     slider.removeEventListener("pointermove", slide);
 }


 function slide(e) {
     slider.style.left = e.clientX;
 }


 const slider = document.getElementById("slider");


 slider.addEventListener("pointerdown", beginSliding);
 slider.addEventListener("pointerup", stopSliding);
````

---

### Initialize UXP Entrypoints Setup

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/entrypoints

This JavaScript code demonstrates the basic initialization of the UXP `entrypoints.setup()` method, which is crucial for handling plugin and panel invocations. It imports the `entrypoints` object from the 'uxp' module.

```javascript
const { entrypoints } = require("uxp");
entrypoints.setup({
	/* ... */
});
```

---

### Example UXP Plugin with Multiple Modules

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/js-modules

This example demonstrates a UXP plugin's main entry point (`index.js`) importing and utilizing functions from separate module files (`video-processor.js` and `ui-helpers.js`). It showcases asynchronous operations and error handling within commands.

```javascript
const { entrypoints } = require("uxp");
const { processVideo } = require("./lib/video-processor.js");
const { showNotification } = require("./lib/ui-helpers.js");

entrypoints.setup({
	commands: {
		processCommand: async () => {
			try {
				await processVideo();
				showNotification("Video processing completed!");
			} catch (error) {
				showNotification("Error: " + error.message);
			}
		},
	},
});
```

```javascript
async function processVideo() {
	// Video processing logic here
	console.log("Processing video...");
	// Simulate async operation
	return new Promise((resolve) => setTimeout(resolve, 1000));
}

function getVideoInfo() {
	// Return video information
	return { duration: 120, format: "mp4" };
}

module.exports = {
	processVideo,
	getVideoInfo,
};
```

```javascript
function showNotification(message) {
	console.log(`Notification: ${message}`);
	// Additional UI notification logic
}

function createProgressBar() {
	// Progress bar creation logic
	return document.createElement("progress");
}

module.exports = {
	showNotification,
	createProgressBar,
};
```

---

### Access User Documents Folder with Full Access using UXP

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

Demonstrates accessing arbitrary file system locations, such as the user's Documents folder, using UXP's `localFileSystem` with the `fullAccess` permission. This requires specifying absolute paths and setting `"localFileSystem": "fullAccess"` in `manifest.json`. The code provides examples for macOS and Windows.

```javascript
const { localFileSystem } = require("uxp").storage;

async function accessUserDocuments() {
	// Access a specific location outside the sandbox
	try {
		// For macOS
		const documentsFolder = await localFileSystem.getEntryWithUrl(
			"file:/Users/user/Documents", // 👈 update with your user
		);
		// For Windows, use: "file:/C:/Users/user/Documents"

		console.log(`Documents folder path: ${documentsFolder.nativePath}`);

		// Read a specific file
		const configFile = await localFileSystem.getEntryWithUrl(
			"file:/Users/user/Documents/config.json", // 👈 update with your user
		);
		if (configFile.isFile) {
			const content = await configFile.read();
			console.log("Config file content:", content);
		}
	} catch (e) {
		console.error("Failed to access documents folder:", e);
	}
}
```

---

### Install Spectrum Web Components (SWC) using npm

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/user-interfaces

Instructions for installing specific Spectrum Web Components (SWC) for use in UXP plugins. It's crucial to lock components to version 0.37.0 for compatibility with the current UXP version.

```bash
npm install @spectrum-web-components/button@0.37.0
npm install @spectrum-web-components/textfield@0.37.0
```

---

### createSetStartAction API

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/media

Creates an action to set the start time of the media.

````APIDOC
## POST /websites/developer_adobe_premiere-pro_uxp/media/actions/setStart

### Description
Returns an action that sets the start time of the media. This method is available from version 25.0 onwards.

### Method
POST

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/media/actions/setStart

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
- **time** (_TickTime_) - The time to set as the start of the media.

### Request Example
```json
{
  "time": "00:00:02:00"
}
````

### Response

#### Success Response (200)

- **action** (_Action_) - An action object that can be executed to set the media start time.

#### Response Example

```json
{
	"action": "setStartAction"
}
```

````

--------------------------------

### Install TypeScript and Create Source Folder

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/typescript-support

Installs the TypeScript compiler as a development dependency and creates a 'src' directory for your TypeScript source files.

```bash
npm install --save-dev typescript
mkdir src
````

---

### Get Entry with URL using LocalFileSystem API

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

This JavaScript code snippet shows how to obtain an `Entry` reference for a file using its URL with the `localFileSystem.getEntryWithUrl()` method. This is a common pattern when working with the `LocalFileSystem` API, which uses object references for file operations.

```javascript
const dataFile = await localFileSystem.getEntryWithUrl("plugin-data:/settings.json");
```

---

### HTML Canvas Setup and JavaScript Interaction

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLCanvasElement

This snippet shows how to set up an HTML canvas element and use JavaScript to retrieve its height, width, and 2D rendering context. It also includes an example of drawing a triangle using the 2D context. This code is intended for use within a UXP environment.

```html
<sp-body id="layers">
	<canvas
		id="canvas"
		height="230"
		width="1500"
	></canvas>
</sp-body>
<footer>
	<sp-button
		id="btnPopulateLoadScript"
		onclick="show_height()"
		>Canvas Height</sp-button
	>
	<sp-button
		id="btnPopulateLoadScript"
		onclick="show_width()"
		>Canvas Width</sp-button
	>
	<sp-button
		id="btnPopulateLoadScript"
		onclick="getContext()"
		>Get Context</sp-button
	>
</footer>
```

```javascript
const canvas = document.getElementById("canvas");

function show_height() {
	console.log("Canvas Height: " + canvas.height);
}

function show_width() {
	console.log("Canvas Width: " + canvas.width);
}

// Function to get the canvas context and draw a triangle using lines
function getContext() {
	let context = canvas.getContext("2d"); // get's the canvas context

	// Draw a triangle. For more details on the below apis refer to interfaces such as CanvasRenderingContext2D, CanvasGradient. The details of the interfaces are shared as a link at the top of this documentation
	context.beginPath();
	context.moveTo(50, 50);
	context.lineTo(100, 50);
	context.lineTo(100, 100);
	context.lineTo(50, 50);
	context.closePath();
	context.stroke();
}
```

---

### Install SWC Button Component using npm

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/swc

Install the Spectrum Web Component for a button into your project using npm. This command fetches the necessary package from the npm registry.

```bash
npm install @swc-uxp-wrappers/button
```

---

### fetch() Network Permissions Limitation Example

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/fetch

This example demonstrates the limitation in UXP v7.4.0 onwards regarding the use of wildcards in top-level domains within the `permissions.network.domains` configuration. It shows incorrect usage with wildcards in TLDs.

```json
"domains": [ "https://www.adobe.*", "https://www.*" ]
```

---

### Access LocalFileSystem and Types API

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

This JavaScript snippet shows how to import the `localFileSystem` and `types` objects from the `uxp` module. These are the primary components for interacting with the file system using the object-oriented `LocalFileSystem` API.

```javascript
const { localFileSystem, types } = require("uxp").storage;
```

---

### UXP Plugin and Panel Setup in JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Entry%20Points/EntryPoints

This snippet demonstrates the basic structure for setting up a UXP plugin and its panels. It includes defining plugin lifecycle methods (create, destroy) and panel-specific methods (create, show, hide, destroy, invokeMenu, update, validateNode). It also shows how to define menu items for panels.

```javascript
const { entrypoints } = require("uxp");

entrypoints.setup({
    plugin: {
        create() { /* ... */ },
        destroy() { /* ... */ }
    },
    panels: {
        "panel1": {
            create() { /* ... */ },
            show() { /* ... */ },
            hide() { /* ... */ },
            destroy() { /* ... */ },
            invokeMenu() { /* ... */ },
            update() { /* ... */ }, // customEntrypoint example
            validateNode() { /* ... */ } // customEntrypoint example
            menuItems: [
                {
                    id: "signIn",
                    label: "Sign In...",
                    enabled: false,
                    checked: false,
                    submenu: [
                        { id: "submenu1", label: "submenu1", enabled: false, checked: false },
                        { "submenu2" }
                    ]
                },
                "-",  // separator.
                "Sign out",  // by default enabled, and the id will be same with the label.
            ]
        },
        "panel2": { /* ... */ }
    },
    commands: {
        "command1": {
            run() { /* ... */ },
            cancel() { /* ... */ }
        },
        "command2": function() { /* ... */ }
    }
});
```

---

### Element Pointer Capture and Event Handling Example (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLDialogElement

Demonstrates how to use setPointerCapture, releasePointerCapture, addEventListener, and removeEventListener to implement a draggable element. This example utilizes pointer events to manage the drag-and-drop functionality.

```javascript
// HTML structure is assumed to be present in the document.
// Example: <div id="slider">SLIDE ME</div>

function beginSliding(e) {
	slider.setPointerCapture(e.pointerId);
	slider.addEventListener("pointermove", slide);
}

function stopSliding(e) {
	slider.releasePointerCapture(e.pointerId);
	slider.removeEventListener("pointermove", slide);
}

function slide(e) {
	slider.style.left = e.clientX + "px"; // Ensure unit is added
}

const slider = document.getElementById("slider");

slider.addEventListener("pointerdown", beginSliding);
slider.addEventListener("pointerup", stopSliding);
```

---

### Send GET Request (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/XMLHttpRequest

Demonstrates how to send a GET request to a server and log the response text. It utilizes the XMLHttpRequest object to handle the request and load event.

```javascript
const xhr = new XMLHttpRequest();
xhr.addEventListener("load", () => {
	console.log(xhr.responseText);
});
xhr.open("GET", "https://www.adobe.com");
xhr.send();
```

---

### Shell Module - Best Practices

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process

Guidelines for using the shell module, including providing clear context, handling user denial, checking platform compatibility, encoding URL parameters, and declaring necessary schemes and extensions.

````APIDOC
## Best Practices

### 1. Provide Clear Context

When using the `developerText` parameter in the `shell.openExternal` method, ensure the explanation is user-friendly and in plain language.

```javascript
// ✅ Good: Clear and specific
await shell.openExternal(
  "https://example.com/guide",
  "Opening tutorial guide in your browser"
);

// ❌ Avoid: Vague or technical
await shell.openExternal(
  "https://example.com/guide",
  "Opening URL"
);
````

### 2. Handle User Denial Gracefully

Users can deny launch requests. Always check the return value of shell methods and provide alternative options if the operation fails or is denied.

```javascript
const result = await shell.openPath(filePath, "Opening project file");
if (result !== "") {
	// User denied or operation failed
	console.log("Unable to open file. Please open it manually.");
}
```

### 3. Check Platform Compatibility

Utilize platform detection to ensure the correct URL schemes are used for different operating systems.

```javascript
const isMac = require("os").platform() === "darwin";
const scheme = isMac ? "maps://" : "bingmaps:";
```

### 4. Encode URL Parameters

When constructing URLs with query parameters, always encode special characters to prevent issues.

```javascript
const subject = encodeURIComponent("My Subject");
const url = `mailto:user@example.com?subject=${subject}`;
```

### 5. Declare All Schemes and Extensions

In your `manifest.json`, only declare the schemes and file extensions that your plugin actively uses to avoid requesting unnecessary permissions.

````

--------------------------------

### Use URL Schemes for File System Access

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

Illustrates the use of UXP's URL schemes to access specific file system locations. `plugin:/`, `plugin-data:/`, and `plugin-temp:/` provide shortcuts to sandbox directories with varying read/write permissions. `file:/` allows access to arbitrary locations but requires `fullAccess` permission.

```html
<img src="plugin:/icons/logo.png" />
<img src="file:/Users/user/Downloads/sample.png" />
<!-- update the path based on your system -->
````

---

### Working with XMP Date and Structs using XMPMeta

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/XMP%20Classes/XMPMeta

Illustrates creating an XMPMeta object and using struct-based APIs, including setting and managing XMPDateTime objects and structure fields. This example requires 'uxp'.

```javascript
let { XMPDateTime, XMPMeta, XMPConst } = require("uxp").xmp;
let meta = new XMPMeta();

let jsDate = new Date("2007-04-10T17:54:50+01:00");
let xmpDate = new XMPDateTime(jsDate);
meta.setProperty(XMPConst.NS_XMP, "CreateDate", xmpDate, XMPConst.XMPDATE);
meta.doesPropertyExist(XMPConst.NS_XMP, "CreateDate");
let prop = meta.getProperty(XMPConst.NS_XMP, "CreateDate", XMPConst.XMPDATE);
meta.deleteProperty(XMPConst.NS_XMP, "CreateDate");

meta.setStructField(XMPConst.NS_XML, "structNameSample", XMPConst.NS_XMP, "sampleFieldName", "sampleFieldValue");
if (meta.doesStructFieldExist(XMPConst.NS_XML, "structNameSample", XMPConst.NS_XMP, "sampleFieldName")) {
	prop = meta.getStructField(XMPConst.NS_XML, "structNameSample", XMPConst.NS_XMP, "sampleFieldName");
	meta.deleteStructField(XMPConst.NS_XML, "structNameSample", XMPConst.NS_XMP, "sampleFieldName");
	if (meta.doesStructFieldExist(XMPConst.NS_XML, "structNameSample", XMPConst.NS_XMP, "sampleFieldName")) {
		console.log("Struct field exists");
	} else {
		console.log("Struct field doesn't exist");
	}
} else {
	console.log("Struct field doesn't exist");
}
```

---

### Shell Module - Summary

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process

An overview of the UXP Shell module, including security considerations for launching processes, the two primary shell methods (`openPath` and `openExternal`), and user experience best practices.

```APIDOC
## Summary

### 1. Launch Process Security

By default, plugins cannot launch external processes. You must declare the `launchProcess` permission in `manifest.json`.

*   **`extensions`**: An array of file extensions your plugin can open (e.g., `[".pdf", ".mp4"]`). Use `""` to allow opening folders.
*   **`schemes`**: An array of URL schemes your plugin can use (e.g., `["https", "mailto"]`).
*   User consent is **always required** for security. Users will see a dialog for each launch attempt unless the choice is remembered.

### 2. Two Shell Methods

*   **`shell.openPath(path, developerText)`**: Opens files or folders in their default system application. Requires the file extension to be listed in `extensions`.
*   **`shell.openExternal(url, developerText)`**: Launches applications via URL schemes. Requires the scheme to be listed in `schemes`. Cannot use `file://` (use `openPath()` instead).

### 3. User Experience Best Practices

*   URL schemes are operating system-specific; use `require("os").platform()` to check the OS and select the appropriate scheme.
*   Provide clear, user-friendly text in the `developerText` parameter.
*   Handle user denial gracefully by checking return values and offering fallback options.
*   Only declare schemes and extensions your plugin actually uses.
*   Encode URL parameters with `encodeURIComponent()` to handle special characters properly.
```

---

### Open User-Selected File with UXP

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

Presents a file picker dialog to the user, allowing them to select a text file to open. It then reads and logs the content of the selected file and its native path. Requires 'request' level permission for localFileSystem.

```javascript
const { localFileSystem, domains, fileTypes } = require("uxp").storage;

async function openUserSelectedFile() {
	// Present a file picker starting at the user's Desktop
	try {
		const file = await localFileSystem.getFileForOpening({
			initialDomain: domains.userDesktop,
			types: fileTypes.text,
		});

		if (!file) {
			console.log("User cancelled file selection");
			return;
		}

		// Read the selected file
		const content = await file.read();
		console.log(`File content:\n${content}`);
		console.log(`File path: ${file.nativePath}`);
	} catch (err) {
		console.error("Failed to open file:", err);
	}
}
```

---

### Get XMLHttpRequest Response URL

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/XMLHttpRequest

The responseURL property returns the URL of the server's response after any redirects. It is an empty string if the URL is null. This example shows how to log the response URL after a GET request.

```javascript
const xhr = new XMLHttpRequest();
xhr.onload = () => {
	console.log(xhr.responseURL);
};
xhr.open("GET", "https://www.adobe.com");
xhr.send();
```

---

### Pointer Interaction Example: Sliding Element

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLHtmlElement

Demonstrates how to use `setPointerCapture` and `releasePointerCapture` along with pointer events to implement a draggable element. This example includes HTML structure and JavaScript event handling for pointer down, move, and up.

```html
<style>
	div {
		width: 140px;
		height: 50px;
		display: flex;
		align-items: center;
		justify-content: center;
		background: #fbe;
		position: absolute;
	}
</style>
<div id="slider">SLIDE ME</div>
```

```javascript
function beginSliding(e) {
	slider.setPointerCapture(e.pointerId);
	slider.addEventListener("pointermove", slide);
}

function stopSliding(e) {
	slider.releasePointerCapture(e.pointerId);
	slider.removeEventListener("pointermove", slide);
}

function slide(e) {
	slider.style.left = e.clientX + "px"; // Added 'px' for correct CSS
}

const slider = document.getElementById("slider");

slider.addEventListener("pointerdown", beginSliding);
slider.addEventListener("pointerup", stopSliding);

// Added pointercancel event for robustness
slider.addEventListener("pointercancel", stopSliding);
```

---

### Get Plugin User ID using UXP JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/User%20Information

This JavaScript code demonstrates how to retrieve the unique GUID of the plugin user using the `userId()` method from the `uxp.userInfo` API. The output is a string representing the user's GUID.

```javascript
let userId = require("uxp").userInfo.userId(); // Get the GUID of plugin user
console.log(userId); // e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

---

### Get Elements by Tag Name (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLScriptElement

Returns a live `NodeList` containing all descendant elements of the parent element that have the specified tag name. The tag name is a string, for example, 'div', 'p', or 'span'.

```javascript
function getElementsByTagName(name) {
	// Implementation details...
}
```

---

### Premiere Pro UXP API Overview

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference

This section provides an overview of how to access the Premiere Pro DOM via UXP. It includes an example of requiring the Premiere Pro module and explains the general capabilities of the API.

````APIDOC
## Overview

The Premiere Pro UXP API allows you to interact with the Premiere Pro DOM. You can open documents, modify them, run menu items, and more.

### Getting Started

To access the Premiere Pro DOM, you need to require the `premierepro` module:

```javascript
const app = require('premierepro');
````

### Key Features

- Access to Premiere Pro objects and methods.
- Ability to manipulate projects and sequences.
- Understanding of synchronous and asynchronous operations.

````

--------------------------------

### Play Media with JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLVideoElement

Initiates media playback for a video element. It returns a resolved Promise upon starting playback and emits a 'play' event. An error event is emitted if playback cannot start.

```javascript
let vid = document.getElementById("sampleVideo");
vid.play();
vid.addEventListener("play", (ev) => {
    console.log("Event - play");
});
````

---

### Pointer Capture Example (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLLinkElement

Demonstrates how to use setPointerCapture and releasePointerCapture to manage pointer events for dragging or interactive elements. Includes HTML and CSS for a draggable element.

```html
<style>
	div {
		width: 140px;
		height: 50px;
		display: flex;
		align-items: center;
		justify-content: center;
		background: #fbe;
		position: absolute;
	}
</style>
<div id="slider">SLIDE ME</div>
```

```javascript
/**
 * Sets pointer capture for the element. This implementation does not dispatch the `gotpointercapture` event on the element.
 * @param {number} pointerId - The unique identifier of the pointer to be captured.
 * @throws {DOMException} If the element is not connected to the DOM.
 */
function setPointerCapture(pointerId) {
	// Implementation details...
}

/**
 * Releases pointer capture for the element. This implementation does not dispatch the `lostpointercapture` event on the element.
 * @param {number} pointerId - The unique identifier of the pointer to be released.
 */
function releasePointerCapture(pointerId) {
	// Implementation details...
}

// Example Usage:
function beginSliding(e) {
	slider.setPointerCapture(e.pointerId);
	slider.addEventListener("pointermove", slide);
}

function stopSliding(e) {
	slider.releasePointerCapture(e.pointerId);
	slider.removeEventListener("pointermove", slide);
}

function slide(e) {
	slider.style.left = e.clientX + "px"; // Ensure units are added
}

const slider = document.getElementById("slider");

slider.addEventListener("pointerdown", beginSliding);
slider.addEventListener("pointerup", stopSliding);
```

---

### HTML h3 Element Example

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-html/Hierarchy/h3

Demonstrates the basic usage of the HTML h3 element in UXP. This element is not theme-aware and requires manual styling for theme adaptation.

```html
<h3>Hello, world!</h3>
```

---

### Get XMLHttpRequest Status Text

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/XMLHttpRequest

The statusText property returns the string associated with the HTTP status code (e.g., 'OK', 'Not Found'). It is an empty string if the request is in the UNSENT or OPENED state. This example logs the status text after a successful response.

```javascript
const xhr = new XMLHttpRequest();
xhr.onload = () => {
	console.log(xhr.statusText);
};
xhr.open("GET", "https://www.adobe.com");
xhr.send();
```

---

### Implement Spectrum Web Component Button

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

Shows how to use a Spectrum Web Component, specifically a button, within a UXP plugin. Requires manual installation, import, and bundling of the component.

```html
<sp-button variant="primary"> Click me </sp-button>
```

---

### Implement Plugin and Panel Lifecycle Hooks in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-lifecycle-hooks

This JavaScript code demonstrates how to implement lifecycle hooks for both the plugin and its panels using the `entrypoints.setup()` method in UXP. It includes examples for plugin creation and destruction, and panel creation, showing, hiding, and destruction. Promises are supported for asynchronous operations. The `rootNode` parameter is available for panel hooks.

```javascript
const { entrypoints } = require("uxp");
entrypoints.setup({
	plugin: {
		create() {
			console.log("Plugin create hook");
		},
		destroy() {
			return new Promise(function (resolve, reject) {
				console.log("Plugin destroy hook");
				resolve();
			});
		},
	},
	panels: {
		firstPanel: {
			// 👈 matches the panel ID from manifest.json
			create(rootNode) {
				return new Promise(function (resolve, reject) {
					console.log("Panel create hook", rootNode);
					resolve();
				});
			},
			show(rootNode, data) {
				return new Promise(function (resolve, reject) {
					console.log("Panel show hook", data);
					resolve();
				});
			},
			hide(rootNode, data) {
				return new Promise(function (resolve, reject) {
					console.log("Panel hide hook", data);
					resolve();
				});
			},
			destroy(rootNode) {
				return new Promise(function (resolve, reject) {
					console.log("Panel destroy hook", rootNode);
					resolve();
				});
			},
		},
	},
});
```

---

### Implement UXP Plugin Lifecycle Hooks

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/entrypoints

This JavaScript code shows how to define 'create' and 'destroy' lifecycle hooks for a UXP plugin using `entrypoints.setup()`. These functions are called when the plugin container is created and destroyed, respectively, allowing for setup and teardown logic.

```javascript
const { entrypoints } = require("uxp");
entrypoints.setup({
	plugin: {
		create() {
			console.log("Plugin created");
		},
		destroy() {
			console.log("Plugin destroyed");
		},
	},
});
```

---

### Instance Methods for Footage Interpretation

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/footageinterpretation

This section covers the instance methods available for getting and setting various footage interpretation properties.

```APIDOC
## Instance Methods

### getAlphaUsage

#### Description
Get alpha usage type property of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_number_) - The alpha usage type.

### getFieldType

#### Description
Get field type of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_number_) - The field type.

### getFrameRate

#### Description
Get frame rate of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_number_) - The frame rate.

### getIgnoreAlpha

#### Description
Get ignore alpha property of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_boolean_) - Whether alpha is ignored.

### getInputLUTID

#### Description
Get input LUTID of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_string_) - The input LUTID.

### getInvertAlpha

#### Description
Get invert alpha property of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_boolean_) - Whether alpha is inverted.

### getPixelAspectRatio

#### Description
Get pixel aspect ratio of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_number_) - The pixel aspect ratio.

### getRemovePullDown

#### Description
Get removePullDown property of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_boolean_) - Whether pull-down is removed.

### getVrConform

#### Description
Get vr conform projection type of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_number_) - The VR conform projection type.

### getVrHorzView

#### Description
Get VR horizontal view of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_number_) - The VR horizontal view value.

### getVrLayout

#### Description
Get VR layout type of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_number_) - The VR layout type.

### getVrVertView

#### Description
Get VR vertical view of footage.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Return Value
*   (_number_) - The VR vertical view value.

### setAlphaUsage

#### Description
Set alpha usage type property of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **alphaUsage** (_number_) - The alpha usage type to set.

### setFieldType

#### Description
Set field type of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **fieldType** (_number_) - The field type to set.

### setFrameRate

#### Description
Set frame rate of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **frameRate** (_number_) - The frame rate to set.

### setIgnoreAlpha

#### Description
Set ignore alpha property of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **ignoreAlpha** (_boolean_) - Whether to ignore alpha.

### setInputLUTID

#### Description
Set input LUTID of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **inputLUTID** (_string_) - The input LUTID to set.

### setInvertAlpha

#### Description
Set invert alpha property of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **invertAlpha** (_boolean_) - Whether to invert alpha.

### setPixelAspectRatio

#### Description
Set pixel aspect ratio of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **pixelAspectRatio** (_number_) - The pixel aspect ratio to set.

### setRemovePullDown

#### Description
Set removePullDown property of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **removePulldown** (_boolean_) - Whether to remove pull-down.

### setVrConform

#### Description
Set VR conform projection type of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **vrConform** (_number_) - The VR conform projection type to set.

### setVrHorzView

#### Description
Set VR horizontal view of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **vrHorzView** (_number_) - The VR horizontal view value to set.

### setVrLayout

#### Description
Set VR layout type of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **vrLayOut** (_number_) - The VR layout type to set.

### setVrVertView

#### Description
Set VR vertical view of footage.

#### Method
SET

#### Endpoint
N/A (Instance Method)

#### Parameters
*   **vrVertView** (_number_) - The VR vertical view value to set.
```

---

### Static Methods

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/project

Methods for project creation, retrieval, and opening.

```APIDOC
## Static Methods

### createProject

#### Description
Create a new project.

#### Parameters
- **path** (string) - The path to create the new project.

### getActiveProject

#### Description
Returns the currently active project.

### getProject

#### Description
Get project referenced by given UID.

#### Parameters
- **projectGuid** (Guid) - The unique identifier of the project to retrieve.

### open

#### Description
Open a project.

#### Parameters
- **path** (string) - The path to the project file.
- **openProjectOptions** (OpenProjectOptions) - Options for opening the project.
```

---

### Combining Spectrum UXP Widgets and SWC in Adobe UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/faqs

This example demonstrates how to seamlessly integrate Spectrum UXP Widgets and Spectrum Web Components (SWC) within an Adobe UXP environment. It showcases the usage of both `<sp-banner>` (SWC) and `<sp-dropdown>` (Spectrum UXP Widget) in the same HTML structure.

```html
<sp-banner>
	<!-- Spectrum Web Components -->
	<div slot="header">Header text</div>
	<div slot="content">Content of the banner</div>
</sp-banner>
<sp-dropdown
	placeholder="Select an option"
	style="width: 320px"
>
	<!-- Spectrum UXP Widget -->
	<sp-menu slot="options">
		<sp-menu-item> Option 1 </sp-menu-item>
		<sp-menu-item> Option 2 </sp-menu-item>
	</sp-menu>
</sp-dropdown>
```

---

### Import UXP XMP Module

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/getting-started

This snippet demonstrates how to import the necessary XMP module from the 'uxp' library. This is the first step required to utilize any of the XMP scripting API functionalities.

```javascript
const xmp = require("uxp").xmp;
```

---

### Styling with :defined in UXP CSS - HTML and CSS Example

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-css/Pseudo-classes/defined

This example demonstrates how to apply styles using the :defined pseudo-class in UXP. It includes HTML for a custom element and a standard paragraph, along with CSS rules that target both elements and specific ones.

```html
<simple-custom text="Custom element example text"></simple-custom>

<p>Standard paragraph example text</p>
```

```css
/* Give the `p` elements distinctive background */
p {
	background: yellow;
}

/* Both the custom and the built-in element are given italic text */
:defined {
	font-style: italic;
}

/* Only simple-custom element is applied with green background*/
simple-custom:defined {
	display: block;
	background: green;
}
```

---

### Select Folder for Export with UXP

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

Allows the user to select a folder for batch export operations. Once a folder is selected, it creates three text files within that folder and writes content to each. Requires 'request' level permission for localFileSystem.

```javascript
const { localFileSystem, domains, fileTypes } = require("uxp").storage;

async function selectFolderForExport() {
	// Let the user select a folder for batch export
	try {
		const folder = await localFileSystem.getFolder({
			initialDomain: domains.userDocuments,
		});

		if (!folder) {
			console.log("User cancelled folder selection");
			return;
		}

		// Create multiple files in the selected folder
		for (let i = 1; i <= 3; i++) {
			const file = await folder.createFile(`export_${i}.txt`, { overwrite: true });
			await file.write(`Content for file ${i}`);
		}

		console.log(`Created 3 files in: ${folder.nativePath}`);
	} catch (err) {
		console.error("Failed to export files:", err);
	}
}
```

---

### WebView Load Start Event Listener (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLWebViewElement

Shows how to attach an event listener to a WebView element to detect when the loading process has begun. The 'loadstart' event provides the URL that is being loaded.

```javascript
const webview = document.getElementById("webviewSample");
// Print the url when loading has started
webview.addEventListener("loadstart", (e) => {
	console.log(`webview.loadstart ${e.url}`);
});
```

---

### Instance Methods

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/project

Methods for managing project state and content.

```APIDOC
## Instance Methods

### close

#### Description
Close a project.

#### Parameters
- **closeProjectOptions** (CloseProjectOptions) - Options for closing the project.

### createSequence

#### Description
Create a new sequence with the default preset path.

#### Parameters
- **name** (string) - The name of the new sequence.
- **presetPath** (string) - Deprecated: use createSequenceWithPresetPath() instead.

### createSequenceFromMedia

#### Description
Create a new sequence with a given name and media.

#### Parameters
- **name** (string) - The name of the new sequence.
- **clipProjectItems** (ClipProjectItem[]) - An array of media items to include in the sequence.
- **targetBin** (ProjectItem) - The bin where the new sequence should be created.

### deleteSequence

#### Description
Delete a given sequence from the project.

#### Parameters
- **sequence** (Sequence) - The sequence to delete.

### executeTransaction

#### Description
Execute an undoable transaction by passing a compound action.

#### Parameters
- **callback** ((compoundAction: CompoundAction) => void) - The callback function containing the compound action.
- **undoString** (string)? - An optional string to describe the transaction for undo history.

### getActiveSequence

#### Description
Get the active sequence of the project.

### getColorSettings

#### Description
Get the project color settings object.

### getInsertionBin

#### Description
Get the current insertion bin.

### getRootItem

#### Description
The root item of the project, which contains all items of the project on the lowest level.

### getSequence

#### Description
Get a sequence by its ID from the project.

#### Parameters
- **guid** (Guid) - The unique identifier of the sequence.

### getSequences

#### Description
Get an array of all sequences in this project.

### importAEComps

#### Description
Import After Effects compositions into the project.

#### Parameters
- **aepPath** (string) - The path to the After Effects project file.
- **compNames** (string[]) - An array of composition names to import.
- **TargetBin** (ProjectItem) - The bin to import the compositions into.

### importAllAEComps

#### Description
Import all After Effects compositions from a project file.

#### Parameters
- **aepPath** (string) - The path to the After Effects project file.
- **TargetBin** (ProjectItem) - The bin to import the compositions into.

### importFiles

#### Description
Import files into the root or target bin of the project.

#### Parameters
- **filePaths** (string[]) - An array of file paths to import.
- **suppressUI** (boolean) - Whether to suppress the user interface during import.
- **targetBin** (ProjectItem) - The bin to import the files into.
- **asNumberedStills** (boolean) - Whether to import files as numbered stills.

### importSequences

#### Description
Import sequences from a project file.

#### Parameters
- **projectPath** (string) - The path to the project file containing the sequences.
- **sequenceIds** (Guid[]) - An array of sequence IDs to import.

### lockedAccess

#### Description
Get read/upgrade locked access to the Project. The project state will not change during the execution of the callback function. Can call executeTransaction while having locked access.

#### Parameters
- **callback** (() => void) - The callback function to execute with locked access.

### openSequence

#### Description
Open a sequence and return true if successful.

#### Parameters
- **sequence** (Sequence) - The sequence to open.

### pauseGrowing

#### Description
Pause growing of files instead of swapping the files.

#### Parameters
- **pause** (boolean) - Whether to pause growing.

### save

#### Description
Save the project.

### saveAs

#### Description
Save the project at the provided path.

#### Parameters
- **path** (string) - The path to save the project to.

### setActiveSequence

#### Description
Set the active sequence of the project.

#### Parameters
- **sequence** (Sequence) - The sequence to set as active.
```

---

### Get Entry Metadata using UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/EntryMetadata

Demonstrates how to retrieve metadata for a plugin folder using the `localFileSystem` and `getMetadata()` method in UXP. This involves accessing the local file system, getting a folder instance, and then logging its name from the metadata.

```javascript
const fs = require("uxp").storage.localFileSystem;
const folder = await fs.getPluginFolder(); // Gets an instance of Folder (or Entry)
const entryMetaData = await folder.getMetadata();
console.log(entryMetaData.name);
```

---

### Element Attribute Handling (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLAnchorElement

Provides methods to get, set, remove, and check for the presence of attributes on an element. This includes getting attribute names, specific attribute nodes, and setting attribute nodes.

```javascript
/**
 * Gets the value of a specified attribute on the element.
 * @param {string} name - Name of the attribute whose value you want to get.
 * @returns {string} The value of the attribute.
 */
function getAttribute(name) {
	// Implementation details would go here
}

/**
 * Sets the value of a specified attribute on the element.
 * @param {string} name - Name of the attribute whose value is to be set.
 * @param {string} value - Value to assign to the attribute.
 */
function setAttribute(name, value) {
	// Implementation details would go here
}

/**
 * Removes a specified attribute from the element.
 * @param {string} name - Name of the attribute to remove.
 */
function removeAttribute(name) {
	// Implementation details would go here
}

/**
 * Checks if the element has a specified attribute.
 * @param {string} name - Name of the attribute to check for.
 * @returns {boolean} True if the attribute exists, false otherwise.
 */
function hasAttribute(name) {
	// Implementation details would go here
}

/**
 * Returns a boolean value indicating whether the current element has any attributes or not.
 * @returns {boolean} True if the element has any attributes, false otherwise.
 */
function hasAttributes() {
	// Implementation details would go here
}

/**
 * Returns the attribute names of the element as an Array of strings.
 * @returns {Array<string>} An array of attribute names.
 */
function getAttributeNames() {
	// Implementation details would go here
}

/**
 * Gets the attribute node with the specified name.
 * @param {string} name - The name of the attribute node to get.
 * @returns {*} The attribute node.
 */
function getAttributeNode(name) {
	// Implementation details would go here
}

/**
 * Sets the attribute node for the element.
 * @param {*} newAttr - The attribute node to set.
 */
function setAttributeNode(newAttr) {
	// Implementation details would go here
}

/**
 * Removes the attribute node from the element.
 * @param {*} oldAttr - The attribute node to remove.
 */
function removeAttributeNode(oldAttr) {
	// Implementation details would go here
}
```

---

### VideoTrack Instance Methods

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/videotrack

Details the instance methods available for the VideoTrack, including getting the track index, media type, track items, and mute state.

```APIDOC
## VideoTrack Instance Methods

### getIndex

#### Description
Returns the index of the track within its track group.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Returns
- **number** - The index representing the track's position.

### getMediaType

#### Description
Returns the unique identifier for the underlying media type of the track.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Returns
- **Guid** - UUID representing the media type.

### getTrackItems

#### Description
Retrieves an array of VideoClipTrackItem objects from the track.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Parameters
##### Query Parameters
- **trackItemType** (Constants.TrackItemType) - Required - Specifies the type of track items to retrieve (e.g., Empty, Clip, Transition, Preview, Feedback).
- **includeEmptyTrackItems** (boolean) - Optional - Whether to include empty track items in the results.

#### Returns
- **VideoClipTrackItem[]** - An array of VideoClipTrackItem objects.

### isMuted

#### Description
Checks if the track is currently muted.

#### Method
GET

#### Endpoint
N/A (Instance Method)

#### Returns
- **boolean** - True if the track is muted, false otherwise.

### setMute

#### Description
Sets the mute state of the track.

#### Method
POST

#### Endpoint
N/A (Instance Method)

#### Parameters
##### Query Parameters
- **mute** (boolean) - Required - Set to true to mute the track, false to unmute.
```

---

### Associate Command Entrypoint with Handler using entrypoints.setup()

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/tutorials/add-commands

This code demonstrates how to associate a command entrypoint with its handler function using the `entrypoints.setup()` method in `index.js`. This approach is suitable for plugins with multiple entrypoint types. Ensure `entrypoints.setup()` is called only once.

```javascript
const { entrypoints } = require("uxp");

function myCommandHandler() {
	console.log("Command invoked!");
}

entrypoints.setup({
	commands: {
		myCommand: myCommandHandler,
	},
});
```

---

### Get File Instance using UXP

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/Persistent%20File%20Storage/File

Demonstrates how to obtain a File instance using the local file system module in UXP. This involves requiring the 'uxp' module and using 'createEntryWithUrl' to get a File object, which can then be checked for its 'isFile' property.

```javascript
const fs = require("uxp").storage.localFileSystem;
const file = await fs.createEntryWithUrl("file:/Users/user/Documents/tmp"); // Gets a File instance
console.log(file.isFile); // returns true
```

---

### Importing Module-Based UXP Core APIs (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/apis

Shows how to import module-based UXP Core APIs using require(). This includes importing the main UXP module, the file system module (fs), and the operating system utilities module (os).

```javascript
// Parent UXP module
const uxp = require("uxp");

// File system access
const fs = require("fs");

// Operating system utilities
const os = require("os");
```

---

### UXP Plugin Manifest Example (JSON)

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

This JSON structure defines the manifest for a UXP plugin. It includes metadata like version, ID, name, host application compatibility, entry points (commands and panels), icons, and network permissions.

```json
{
	"manifestVersion": 5,
	"id": "YOUR_ID_HERE",
	"name": "Name of your plugin",
	"version": "1.0.0",
	"main": "index.html",
	"host": {
		"app": "HOST_APPLICATION",
		"minVersion": "HOST_VERSION"
	},
	"entrypoints": [
		{
			"type": "command",
			"id": "commandFn",
			"label": {
				"default": "Show A Dialog"
			}
		},
		{
			"type": "panel",
			"id": "panelName",
			"label": {
				"default": "Panel Name"
			}
		}
	],
	"icons": [
		{
			"width": 24,
			"height": 24,
			"path": "icons/icon.png",
			"scale": [1, 2]
		}
	],
	"requiredPermissions": {
		"network": {
			"domains": "all"
		}
	}
}
```

---

### Enable User Info Access in UXP

Source: https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest

This snippet shows how to enable and access anonymized user GUID information in UXP plugins. It requires setting 'enableUserInfo' to true in the manifest and then using 'uxp.userInfo.userId()' to retrieve the GUID.

```javascript
// set "enableUserInfo" to true in the manifest

let userId = require("uxp").userInfo.userId(); // Get the GUID of user
console.log(userId);
// "dad8483a3682a0c3e0fa990281142353901a69fc371254edde8b7dd38ca604c6"
```

---

### Applying Styling with UXP and Premiere APIs (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/apis

This example outlines the conceptual use of both Premiere APIs for modifying clip properties and UXP APIs for UI feedback, such as showing progress, when applying styles to elements within Premiere Pro.

```javascript
// Premiere APIs to modify the clip properties
// UXP APIs to show progress in your plugin's UI
```

---

### Read and Write Files in Plugin Sandbox with UXP 'fs' Module (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

This snippet demonstrates how to use the 'fs' module, which provides a Node.js-like interface for file operations within the UXP plugin environment. It shows how to read a configuration file from the plugin's directory using the 'plugin:/' URL scheme and how to write data to the plugin's data folder using 'plugin-data:/'.

```javascript
const fs = require("fs");

async function readConfigFile() {
	// Read a configuration file from the plugin folder
	try {
		const content = await fs.readFile("plugin:/config.json", "utf8");
		const config = JSON.parse(content);
		console.log("Configuration loaded:", config);
	} catch (e) {
		console.error("Failed to read config file:", e);
	}
}

async function writeToDataFolder() {
	// Write data to the plugin's data folder
	try {
		const data = {
			lastRun: new Date().toISOString(),
			version: "1.0.0",
		};

		await fs.writeFile("plugin-data:/state.json", JSON.stringify(data, null, 2), "utf-8");

		console.log("State saved successfully");
	} catch (e) {
		console.error("Failed to save state:", e);
	}
}
```

---

### Mixing HTML, SWC, and Spectrum UXP Widgets

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum

Provides an example of seamlessly integrating plain HTML elements, Spectrum Web Components (SWC), and built-in Spectrum UXP widgets within a single form structure.

```html
<form>
	<sp-banner>
		<div slot="header">Header text</div>
		<div slot="content">Content of the banner</div>
	</sp-banner>
	<sp-button variant="primary">I'm a button</sp-button>
</form>
```

---

### Configure Local File System Access (JSON)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

This JSON snippet configures the necessary permissions for a UXP plugin to access the local file system. It specifies 'localFileSystem' with 'fullAccess', allowing the plugin to read and write to any location on the user's file system.

```json
{
	"manifestVersion": 5,
	// ...
	"requiredPermissions": {
		"localFileSystem": "fullAccess"
	}
	// ...
}
```

---

### Media Properties API

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/media

Access and retrieve properties of the media, including its start time and duration.

````APIDOC
## Media Properties API

### Description
This API provides access to the properties of a media item, such as its start time and duration.

### Method
GET

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/media

### Parameters
#### Path Parameters
None

#### Query Parameters
None

### Request Example
```json
{}
````

### Response

#### Success Response (200)

- **start** (_TickTime_) - The start time of the media.
- **duration** (_TickTime_) - The duration of the media.

#### Response Example

```json
{
	"start": "00:00:01:00",
	"duration": "00:00:10:00"
}
```

````

--------------------------------

### Reading a File using UXP Core APIs (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/apis

This snippet demonstrates how to read content from a file using UXP Core APIs. It uses the 'fs' module to get a file object for opening and then reads its content asynchronously.

```javascript
const fs = require("fs");
const file = await fs.getFileForOpening();
const content = await file.read();
````

---

### Create File in Folder using UXP LocalFileSystem

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/filesystem-operations

Demonstrates how to create a new file within a specified folder using UXP's `localFileSystem`. It includes error handling and checks if the entry is indeed a folder before creating the file. Requires the `uxp` module.

```javascript
const { localFileSystem, types } = require("uxp").storage;

async function createFileInFolder() {
	try {
		// Create a folder entry
		const folderEntry = await localFileSystem.createEntryWithUrl("plugin-temp:/myFolder", { type: types.folder });

		// Verify it's a folder before using folder-specific methods
		if (folderEntry.isFolder) {
			const newFile = await folderEntry.createFile("data.txt", { overwrite: true });
			await newFile.write("This is sample content.");
			console.log(`File created at: ${newFile.nativePath}`);
		}
	} catch (e) {
		console.error("Failed to create file:", e);
	}
}
```

---

### Log Host Environment Information with UXP and Node.js

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/host-info

This snippet demonstrates how to log detailed host environment information, including OS, application name and version, UXP runtime version, plugin version, and UI locale. It requires the `uxp` and `os` modules. The output is logged to the console.

```javascript
const { host, versions } = require("uxp");
const os = require("os");

// Log host environment information 💻
function logHostInfo() {
	console.log("=== Host Environment ===");
	console.log(`OS: ${os.platform()} ${os.release()}`);
	console.log(`Application: ${host.name} v${host.version}`);
	console.log(`UXP Runtime: v${versions.uxp}`);
	console.log(`Plugin Version: v${versions.plugin}`);
	console.log(`UI Locale: ${host.uiLocale}`);
}
logHostInfo();
```

---

### Get File/Folder Stats Asynchronously - JavaScript

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/fs

Gets information asynchronously about a file or folder at a given path. The returned object contains statistics similar to Node.js's `Stats` class, though some properties might be platform-dependent. If no callback is provided, it returns a Promise.

```javascript
const stats = await fs.lstat("plugin-data:/textFile.txt");
const isFile = stats.isFile();
```

---

### getNamespacePrefix - Get Namespace Prefix

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Modules/uxp/XMP/XMP%20Classes/XMPMeta

Retrieves the prefix associated with a registered namespace URI.

````APIDOC
## getNamespacePrefix(namespaceURI)

### Description
Retrieves the prefix associated with a registered namespace URI.

### Method
`getNamespacePrefix`

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
* **namespaceURI** (string) - Required - The namespace URI string. Refer to Schema namespace string constants.

### Request Example
```javascript
XMPMeta.getNamespacePrefix(namespaceURI)
````

### Response

#### Success Response (200)

- **Return Value** (string) - The prefix string, followed by a colon.

#### Response Example

```json
{
	"return": "prefix:"
}
```

````

--------------------------------

### ProjectSettings - createSetScratchDiskSettingsAction

Source: https://developer.adobe.com/premiere-pro/uxp/ppro_reference/classes/projectsettings

Returns an action which sets ScratchDiskSettings for a project.

```APIDOC
## POST /websites/developer_adobe_premiere-pro_uxp/ProjectSettings/createSetScratchDiskSettingsAction

### Description
Returns an action which sets ScratchDiskSettings for a project.

### Method
POST

### Endpoint
/websites/developer_adobe_premiere-pro_uxp/ProjectSettings/createSetScratchDiskSettingsAction

### Parameters
#### Request Body
- **project** (_Project_) - Required - The project for which to set scratch disk settings.
- **scratchDiskSettings** (_ScratchDiskSettings_) - Required - The scratch disk settings to apply.

### Request Example
{
  "project": "project_object",
  "scratchDiskSettings": "scratch_disk_settings_object"
}

### Response
#### Success Response (200)
- **action** (_Action_) - An action object that can be invoked to set the scratch disk settings.

#### Response Example
{
  "action": "action_object"
}
````

---

### Headers Manipulation Methods

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/Headers

Methods for appending, deleting, getting, and setting header values.

```APIDOC
## append(name, value)

### Description
Appends a new value onto an existing header or adds the header if it does not exist.

### Method
`append`

### Parameters
- **name** (`string`) - Required - The name of the HTTP header.
- **value** (`string`) - Required - The value of the HTTP header.
```

```APIDOC
## delete(name)

### Description
Deletes a header from the current Header object.

### Method
`delete`

### Parameters
- **name** (`string`) - Required - The name of the HTTP header.
```

```APIDOC
## get(name)

### Description
Returns a byte string of all values of a header within the Headers object. Returns null if the header does not exist.

### Method
`get`

### Returns
- `string` - The value of the retrieved header, or null if not found.

### Parameters
- **name** (`string`) - Required - The name of the HTTP header.
```

```APIDOC
## set(name, value)

### Description
Sets a new value for an existing header or adds the header if it does not exist.

### Method
`set`

### Parameters
- **name** (`string`) - Required - The name of the HTTP header.
- **value** (`string`) - Required - The value of the HTTP header.
```

---

### fetch() Method Signature and Parameters

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/Data%20Transfers/fetch

This documentation outlines the signature and parameters for the window.fetch() method. It details the 'input' argument (URL string or Request object) and the optional 'init' object, which allows customization of HTTP request methods, headers, body, and credentials.

```javascript
window.fetch(input, [init])

Parameters:
- input: `string` or `Request` - The URL string or a Request object.
- [init]: `Object` (Optional) - Custom settings for the HTTP request.
  - [init.method]: `string` - HTTP request method (default: "GET").
  - [init.headers]: `Headers` - HTTP request headers.
  - [init.body]: `string` | `ArrayBuffer` | `TypedArray` | `Blob` | `FormData` | `URLSearchParams` - Request body.
  - [init.credentials]: `string` - Indicates whether to send cookies ("omit" or "include").
```

---

### Attribute Manipulation

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLHtmlElement

Methods for getting, setting, and checking element attributes.

````APIDOC
## getAttribute(name)

### Description
Returns the value of a specified attribute on the element.

### Method
GET

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage
// const value = element.getAttribute('data-custom');
````

### Response

#### Success Response (200)

- **string** - The value of the attribute, or null if the attribute is not set.

#### Response Example

```json
"some-value"
```

````

```APIDOC
## setAttribute(name, value)

### Description
Sets the value of a specified attribute on the element. If the attribute already exists, its value is updated; otherwise, a new attribute is added.

### Method
PUT (conceptually, as it modifies state)

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage
// element.setAttribute('data-custom', 'new-value');
````

### Response

#### Success Response (200)

None (modifies element state)

#### Response Example

None

````

```APIDOC
## removeAttribute(name)

### Description
Removes a specified attribute from the element.

### Method
DELETE (conceptually, as it removes state)

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage
// element.removeAttribute('data-custom');
````

### Response

#### Success Response (200)

None (modifies element state)

#### Response Example

None

````

```APIDOC
## hasAttribute(name)

### Description
Returns a boolean indicating whether the element has the specified attribute.

### Method
GET

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage
// const hasAttr = element.hasAttribute('data-custom'); // true or false
````

### Response

#### Success Response (200)

- **boolean** - True if the attribute exists, false otherwise.

#### Response Example

```json
true
```

````

```APIDOC
## hasAttributes()

### Description
Returns a boolean value indicating whether the current element has any attributes or not.

### Method
GET

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage
// const hasAnyAttributes = element.hasAttributes(); // true or false
````

### Response

#### Success Response (200)

- **boolean** - True if the element has any attributes, false otherwise.

#### Response Example

```json
false
```

````

```APIDOC
## getAttributeNames()

### Description
Returns the attribute names of the element as an Array of strings.

### Method
GET

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage
// const attributeNames = element.getAttributeNames(); // ['id', 'class', ...]
````

### Response

#### Success Response (200)

- **Array** - An array of strings, where each string is an attribute name.

#### Response Example

```json
["id", "class", "data-info"]
```

````

```APIDOC
## getAttributeNode(name)

### Description
Returns the attribute node object for the specified attribute.

### Method
GET

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage
// const attrNode = element.getAttributeNode('id');
````

### Response

#### Success Response (200)

- **\* ** - The attribute node object, or null if the attribute is not found.

#### Response Example

```json
{
	"name": "id",
	"value": "myElement"
}
```

````

```APIDOC
## setAttributeNode(newAttr)

### Description
Adds or replaces a new attribute node in the element.

### Method
PUT (conceptually, as it modifies state)

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage
// const newAttr = document.createAttribute('data-custom');
// newAttr.value = 'example';
// element.setAttributeNode(newAttr);
````

### Response

#### Success Response (200)

- **\* ** - The previous attribute node if it was replaced, otherwise null.

#### Response Example

```json
null
```

````

```APIDOC
## removeAttributeNode(oldAttr)

### Description
Removes an attribute node from the element.

### Method
DELETE (conceptually, as it removes state)

### Endpoint
N/A (Method call on an element instance)

### Parameters
#### Path Parameters
None

#### Query Parameters
None

#### Request Body
None

### Request Example
```javascript
// Example usage
// const attrNodeToRemove = element.getAttributeNode('data-custom');
// if (attrNodeToRemove) {
//   element.removeAttributeNode(attrNodeToRemove);
// }
````

### Response

#### Success Response (200)

- **\* ** - The removed attribute node.

#### Response Example

```json
{
	"name": "data-custom",
	"value": "example"
}
```

````

--------------------------------

### Shell Module - Troubleshooting Common Issues

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process

Solutions for common problems encountered when using the shell module, including permission errors, silent failures, platform-specific scheme issues, and file access restrictions.

```APIDOC
## Troubleshoot Common Issues

| Symptom                       | Likely Cause                      | Solution                                                     |
|-------------------------------|-----------------------------------|--------------------------------------------------------------|
| Permission denied error       | Missing manifest entry            | Add the extension or scheme to `launchProcess` in manifest. |
| Operation fails silently      | User denied consent               | Check return value and handle denial gracefully.             |
| Platform-specific scheme not working | Wrong scheme for OS             | Use platform detection to choose the correct scheme.         |
| `file://` scheme doesn't work | Wrong method used                 | Use `openPath()` for local files, not `openExternal()`.      |
| UWP restrictions on Windows   | System security policy            | UWP apps can only access files in their sandbox.            |
````

---

### Element Attribute Manipulation

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLButtonElement

A set of methods for getting, setting, and removing attributes from an HTML element. Includes methods to retrieve attribute values by name, set or update attributes, remove attributes, check for attribute existence, get all attribute names, and manage attribute nodes.

```javascript
/**
 * Returns the value of the specified attribute on the element.
 * @param {string} name - Name of the attribute whose value you want to get.
 * @returns {string} The value of the attribute.
 */
function getAttribute(name) {
	// Implementation details...
	return ""; // Placeholder
}

/**
 * Sets the value of a specified attribute on the element.
 * @param {string} name - Name of the attribute whose value is to be set.
 * @param {string} value - Value to assign to the attribute.
 */
function setAttribute(name, value) {
	// Implementation details...
}

/**
 * Removes the specified attribute from the element.
 * @param {string} name - Name of the attribute to remove.
 */
function removeAttribute(name) {
	// Implementation details...
}

/**
 * Returns a boolean value indicating whether the element has the specified attribute.
 * @param {string} name - Name of the attribute to check for.
 * @returns {boolean} True if the attribute exists, false otherwise.
 */
function hasAttribute(name) {
	// Implementation details...
	return false; // Placeholder
}

/**
 * Returns a boolean value indicating whether the current element has any attributes or not.
 * @returns {boolean} True if the element has any attributes, false otherwise.
 */
function hasAttributes() {
	// Implementation details...
	return false; // Placeholder
}

/**
 * Returns an array of strings representing all attribute names of the element.
 * @returns {Array<string>} An array of attribute names.
 */
function getAttributeNames() {
	// Implementation details...
	return []; // Placeholder
}

/**
 * Returns the attribute node with the specified name.
 * @param {string} name - The name of the attribute node to retrieve.
 * @returns {Attr | null} The attribute node or null if not found.
 */
function getAttributeNode(name) {
	// Implementation details...
	return null; // Placeholder
}

/**
 * Sets a new attribute node to the element.
 * @param {Attr} newAttr - The attribute node to set.
 * @returns {Attr | null} The previous attribute node if one existed, otherwise null.
 */
function setAttributeNode(newAttr) {
	// Implementation details...
	return null; // Placeholder
}

/**
 * Removes the specified attribute node from the element.
 * @param {Attr} oldAttr - The attribute node to remove.
 * @returns {Attr | null} The removed attribute node or null if not found.
 */
function removeAttributeNode(oldAttr) {
	// Implementation details...
	return null; // Placeholder
}
```

---

### Interactive Element Dragging Example (JavaScript)

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLLabelElement

Demonstrates how to implement drag-and-drop functionality for an HTML element using pointer events, setPointerCapture, and releasePointerCapture. This allows users to click and drag an element within the viewport.

```javascript
// HTML
/*
<style>
    div {
        width: 140px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #fbe;
        position: absolute;
    }
</style>
<div id="slider">SLIDE ME</div>
*/

// JS
function beginSliding(e) {
	slider.setPointerCapture(e.pointerId);
	slider.addEventListener("pointermove", slide);
}

function stopSliding(e) {
	slider.releasePointerCapture(e.pointerId);
	slider.removeEventListener("pointermove", slide);
}

function slide(e) {
	// Note: In a real scenario, you might want to calculate deltaX/deltaY
	// and adjust based on the initial click position for smoother dragging.
	slider.style.left = e.clientX + "px"; // Using clientX for simplicity
}

const slider = document.getElementById("slider");

slider.addEventListener("pointerdown", beginSliding);
slider.addEventListener("pointerup", stopSliding);

// Adding pointercancel is good practice for robustness
slider.addEventListener("pointercancel", stopSliding);
```

---

### Attribute Manipulation API

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLDialogElement

APIs for getting, setting, and checking element attributes.

```APIDOC
## getAttribute(name)

### Description
Returns the value of a specified attribute on the element.

### Method
`getAttribute`

### Parameters
#### Path Parameters
- **name** (string) - The name of the attribute whose value you want to get.

### Returns
`string`

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/getAttribute

---

## setAttribute(name, value)

### Description
Sets or modifies the value of a specified attribute on the element.

### Method
`setAttribute`

### Parameters
#### Path Parameters
- **name** (string) - The name of the attribute whose value is to be set.
- **value** (string) - The value to assign to the attribute.

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/setAttribute

---

## removeAttribute(name)

### Description
Removes a specified attribute from the element.

### Method
`removeAttribute`

### Parameters
#### Path Parameters
- **name** (string) - The name of the attribute to remove.

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/removeAttribute

---

## hasAttribute(name)

### Description
Returns a boolean indicating whether the element has the specified attribute.

### Method
`hasAttribute`

### Parameters
#### Path Parameters
- **name** (string) - The name of the attribute to check for.

### Returns
`boolean`

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/hasAttribute

---

## hasAttributes()

### Description
Returns a boolean value indicating whether the current element has any attributes or not.

### Method
`hasAttributes`

### Parameters
None

### Returns
`boolean`

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/hasAttributes

---

## getAttributeNames()

### Description
Returns the attribute names of the element as an Array of strings.

### Method
`getAttributeNames`

### Parameters
None

### Returns
`Array`

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/getAttributeNames

---

## getAttributeNode(name)

### Description
Returns the attribute node with the specified name.

### Method
`getAttributeNode`

### Parameters
#### Path Parameters
- **name** (string) - The name of the attribute node to retrieve.

### Returns
`*`

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/getAttributeNode

---

## setAttributeNode(newAttr)

### Description
Adds or replaces a specified attribute node in the element.

### Method
`setAttributeNode`

### Parameters
#### Path Parameters
- **newAttr** (*) - The attribute node to add or replace.

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/setAttributeNode

---

## removeAttributeNode(oldAttr)

### Description
Removes a specified attribute node from the element.

### Method
`removeAttributeNode`

### Parameters
#### Path Parameters
- **oldAttr** (*) - The attribute node to remove.

### See Also
- https://developer.mozilla.org/en-US/docs/Web/API/Element/removeAttributeNode
```

---

### Render sp-menu-item Components in HTML

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-spectrum/Spectrum%20UXP%20Widgets/User%20Interface/sp-menu-item

Demonstrates how to render basic, selected, and disabled menu items using the sp-menu-item component. This is a fundamental usage example for creating menu options.

```html
<sp-menu-item>Chicago</sp-menu-item>
<sp-menu-item selected>New York City</sp-menu-item>
<sp-menu-item disabled>St. Louis</sp-menu-item>
```

---

### Element Attribute Management

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLSelectElement

Methods for getting, setting, and removing element attributes.

```APIDOC
## getAttribute(name)

### Description
Gets the value of a specified attribute on the element.

### Method
`getAttribute`

### Parameters
#### Path Parameters
- **name** (string) - Name of the attribute whose value you want to get.

### Returns
`string`

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/getAttribute
```

```APIDOC
## setAttribute(name, value)

### Description
Sets the value of a specified attribute on the element.

### Method
`setAttribute`

### Parameters
#### Path Parameters
- **name** (string) - Name of the attribute whose value is to be set.
- **value** (string) - Value to assign to the attribute.

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/setAttribute
```

```APIDOC
## removeAttribute(name)

### Description
Removes a specified attribute from the element.

### Method
`removeAttribute`

### Parameters
#### Path Parameters
- **name** (string) - Description not available

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/removeAttribute
```

```APIDOC
## hasAttribute(name)

### Description
Checks if the element has a specified attribute.

### Method
`hasAttribute`

### Parameters
#### Path Parameters
- **name** (string) - Description not available

### Returns
`boolean`

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/hasAttribute
```

```APIDOC
## hasAttributes()

### Description
Returns a boolean value indicating whether the current element has any attributes or not.

### Method
`hasAttributes`

### Returns
`boolean`

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/hasAttributes
```

```APIDOC
## getAttributeNames()

### Description
Returns the attribute names of the element as an Array of strings.

### Method
`getAttributeNames`

### Returns
`Array`

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/getAttributeNames
```

```APIDOC
## getAttributeNode(name)

### Description
Gets the attribute node with the specified name.

### Method
`getAttributeNode`

### Parameters
#### Path Parameters
- **name** (string) - Description not available

### Returns
`*`

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/getAttributeNode
```

```APIDOC
## setAttributeNode(newAttr)

### Description
Sets a new attribute node on the element.

### Method
`setAttributeNode`

### Parameters
#### Path Parameters
- **newAttr** (*) - Description not available

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/setAttributeNode
```

```APIDOC
## removeAttributeNode(oldAttr)

### Description
Removes an attribute node from the element.

### Method
`removeAttributeNode`

### Parameters
#### Path Parameters
- **oldAttr** (*) - Description not available

### See
https://developer.mozilla.org/en-US/docs/Web/API/Element/removeAttributeNode
```

---

### Access UXP Shell API

Source: https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process

Import the `shell` object from the `uxp` module to access functionalities for interacting with the operating system's shell.

```javascript
const { shell } = require("uxp");
```

---

### Attribute Manipulation

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLHeadElement

Methods for getting, setting, and removing attributes from an element.

````APIDOC
## getAttribute(name)

### Description
Retrieves the value of a specified attribute from the element.

### Method
N/A (JavaScript method)

### Endpoint
N/A

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Parameters
#### Path Parameters
N/A

#### Query Parameters
N/A

#### Request Body
N/A

### Request Example
```javascript
// const attributeValue = element.getAttribute('data-custom');
````

### Response

#### Success Response (200)

Returns the attribute's value as a string, or null if the attribute is not set.

#### Response Example

N/A

## setAttribute(name, value)

### Description

Sets or updates the value of a specified attribute on the element.

### Method

N/A (JavaScript method)

### Endpoint

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Request Example

```javascript
// element.setAttribute('class', 'new-class');
```

### Response

#### Success Response (200)

N/A

#### Response Example

N/A

## removeAttribute(name)

### Description

Removes a specified attribute from the element.

### Method

N/A (JavaScript method)

### Endpoint

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Request Example

```javascript
// element.removeAttribute('disabled');
```

### Response

#### Success Response (200)

N/A

#### Response Example

N/A

## hasAttribute(name)

### Description

Checks if the element has a specified attribute.

### Method

N/A (JavaScript method)

### Endpoint

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Request Example

```javascript
// const hasDataAttribute = element.hasAttribute('data-id');
```

### Response

#### Success Response (200)

Returns `true` if the attribute exists, `false` otherwise.

#### Response Example

N/A

## hasAttributes()

### Description

Returns a boolean value indicating whether the current element has any attributes or not.

### Method

N/A (JavaScript method)

### Endpoint

N/A

### Parameters

N/A

### Request Example

```javascript
// const hasAnyAttributes = element.hasAttributes();
```

### Response

#### Success Response (200)

Returns `true` if the element has at least one attribute, `false` otherwise.

#### Response Example

N/A

## getAttributeNames()

### Description

Returns the attribute names of the element as an Array of strings.

### Method

N/A (JavaScript method)

### Endpoint

N/A

### Parameters

N/A

### Request Example

```javascript
// const attributeNames = element.getAttributeNames();
```

### Response

#### Success Response (200)

Returns an Array of strings, where each string is an attribute name.

#### Response Example

```json
{
	"example": ["id", "class", "data-custom"]
}
```

## getAttributeNode(name)

### Description

Retrieves the attribute node with the specified name.

### Method

N/A (JavaScript method)

### Endpoint

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Request Example

```javascript
// const attributeNode = element.getAttributeNode('style');
```

### Response

#### Success Response (200)

Returns the attribute node, or null if not found.

#### Response Example

N/A

## setAttributeNode(newAttr)

### Description

Adds a new attribute node to the element. If an attribute with the same name already exists, it is replaced.

### Method

N/A (JavaScript method)

### Endpoint

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Request Example

```javascript
// const newAttribute = document.createAttribute('data-info');
// newAttribute.value = 'some value';
// element.setAttributeNode(newAttribute);
```

### Response

#### Success Response (200)

Returns the attribute node that was added or replaced.

#### Response Example

N/A

## removeAttributeNode(oldAttr)

### Description

Removes an attribute node from the element.

### Method

N/A (JavaScript method)

### Endpoint

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Parameters

#### Path Parameters

N/A

#### Query Parameters

N/A

#### Request Body

N/A

### Request Example

```javascript
// const attributeToRemove = element.getAttributeNode('data-info');
// if (attributeToRemove) {
//   element.removeAttributeNode(attributeToRemove);
// }
```

### Response

#### Success Response (200)

Returns the removed attribute node.

#### Response Example

N/A

````

--------------------------------

### Attribute Manipulation

Source: https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/Global%20Members/HTML%20Elements/HTMLButtonElement

Methods for getting, setting, and removing attributes from an element.

```APIDOC
## getAttribute(name)

### Description
Returns the value of a specified attribute on the element.

### Method

[Implicitly defined by function signature]

### Endpoint

[Not applicable for this method]

### Parameters
#### Path Parameters

[None]

#### Query Parameters

[None]

#### Request Body

[None]

### Request Example

```javascript
// const className = element.getAttribute('class');
````

### Response

#### Success Response (200)

- **Attribute Value** (`string`) - The value of the attribute, or `null` if the attribute is not set.

#### Response Example

```json
{
	"value": "some-class"
}
```

## setAttribute(name, value)

### Description

Sets or changes the value of a specified attribute on the element.

### Method

[Implicitly defined by function signature]

### Endpoint

[Not applicable for this method]

### Parameters

#### Path Parameters

[None]

#### Query Parameters

[None]

#### Request Body

[None]

### Request Example

```javascript
// element.setAttribute('id', 'new-id');
```

### Response

#### Success Response (200)

[No explicit return value, behavior is side-effectual]

#### Response Example

[None]

## removeAttribute(name)

### Description

Removes a specified attribute from the element.

### Method

[Implicitly defined by function signature]

### Endpoint

[Not applicable for this method]

### Parameters

#### Path Parameters

[None]

#### Query Parameters

[None]

#### Request Body

[None]

### Request Example

```javascript
// element.removeAttribute('data-custom');
```

### Response

#### Success Response (200)

[No explicit return value, behavior is side-effectual]

#### Response Example

[None]

## hasAttribute(name)

### Description

Returns a boolean indicating whether the element has the specified attribute.

### Method

[Implicitly defined by function signature]

### Endpoint

[Not applicable for this method]

### Parameters

#### Path Parameters

[None]

#### Query Parameters

[None]

#### Request Body

[None]

### Request Example

```javascript
// const hasId = element.hasAttribute('id');
```

### Response

#### Success Response (200)

- **Has Attribute** (`boolean`) - `true` if the attribute exists, `false` otherwise.

#### Response Example

```json
{
	"hasAttribute": true
}
```

## hasAttributes()

### Description

Returns a boolean value indicating whether the current element has any attributes or not.

### Method

[Implicitly defined by function signature]

### Endpoint

[Not applicable for this method]

### Parameters

#### Path Parameters

[None]

#### Query Parameters

[None]

#### Request Body

[None]

### Request Example

```javascript
// const hasAnyAttributes = element.hasAttributes();
```

### Response

#### Success Response (200)

- **Has Attributes** (`boolean`) - `true` if the element has one or more attributes, `false` otherwise.

#### Response Example

```json
{
	"hasAttributes": true
}
```

## getAttributeNames()

### Description

Returns the attribute names of the element as an Array of strings.

### Method

[Implicitly defined by function signature]

### Endpoint

[Not applicable for this method]

### Parameters

#### Path Parameters

[None]

#### Query Parameters

[None]

#### Request Body

[None]

### Request Example

```javascript
// const attributeNames = element.getAttributeNames();
```

### Response

#### Success Response (200)

- **Attribute Names** (`Array<string>`) - An array containing the names of all attributes of the element.

#### Response Example

```json
{
	"attributeNames": ["id", "class", "data-custom"]
}
```
