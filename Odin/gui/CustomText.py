import tkinter as tk
import tkinter.filedialog as filedialog
import pyperclip
import sys

class CustomText(tk.Text):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self.configure(
            bg="#282828",
            fg="#FFFFFF",
            insertbackground="#FFFFFF",
            padx=0,
            pady=0,
            highlightthickness=0,
            relief='flat',
            wrap='word'
        )

        self.tag_configure(
            "user",
            background="#343434",
            foreground="#FFFFFF",
            font=("Segoe UI", 12),
            lmargin1=50,
            lmargin2=50,
            rmargin=100,
            wrap='word',
            spacing1=10,
            spacing2=0,
            spacing3=10
        )

        self.tag_configure(
            "assistant",
            background="#1E1E1E",
            foreground="#FFFFFF",
            font=("Segoe UI", 12),
            lmargin1=50,
            lmargin2=50,
            rmargin=100,
            wrap='word',
            spacing1=10,
            spacing2=0,
            spacing3=10
        )

        self.tag_configure(
            "code",
            background="#000000",
            foreground="#FFFFFF",
            font=("Courier New", 12),
            lmargin1=50,
            lmargin2=50,
            rmargin=100,
            wrap='word',
            spacing1=10,
            spacing2=0,
            spacing3=10
        )

        self.tag_configure(
            "buttons",
            background="#000000",
            foreground="#0078D7",
            font=("Courier New", 12, "underline"),
            lmargin1=50,
            lmargin2=50,
            rmargin=100,
            wrap='none',
            spacing1=10,
            spacing2=0,
            spacing3=10
        )

        self.tag_bind("buttons", "<Button-1>", self.handle_button_click)
        self.tag_bind("buttons", "<Enter>", self.on_button_enter)
        self.tag_bind("buttons", "<Leave>", self.on_button_leave)

    def handle_button_click(self, event):
        index = self.index("@%s,%s" % (event.x, event.y))
        print(f"Clicked at index: {index}")

        code_start, code_end = self.tag_prevrange("code", index)
        if code_start and code_end:
            code_content = self.get(code_start, code_end).strip()
            print(f"Code content: {code_content}")

            clicked_word = self.get(index, index + " wordend")
            print(f"Clicked word: {clicked_word}")
            if "Copy" in clicked_word:
                self.copy_code_to_clipboard(code_content)
            elif "Save" in clicked_word:
                self.save_code_to_file(code_content)
        else:
            print("Error: Could not find the adjacent code block.")

    def copy_code_to_clipboard(self, code_content):
        pyperclip.copy(code_content)
        print("Codeblock Copied!")

    def save_code_to_file(self, code_content):
        file_path = filedialog.asksaveasfilename(defaultextension=".py", filetypes=[("Python Files", "*.py"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, "w") as file:
                file.write(code_content)
            print("Codeblock Saved!")

    def insert_code(self, code, language=None):
        self.insert('end', code + "\n", "code")
        self.insert('end', "[Copy] [Save]\n", ("buttons",))
        self.mark_set("insert", "end-2c")

    def on_button_enter(self, event):
        self.config(cursor="hand2")

    def on_button_leave(self, event):
        self.config(cursor="")